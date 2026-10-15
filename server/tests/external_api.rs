#[path = "support/fake_provider.rs"]
mod fake_provider;
#[path = "support/fixtures.rs"]
mod fixtures;

use std::{collections::HashSet, sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::{header, HeaderMap, Request, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use cursor_server::{
    api::byok,
    control,
    model::{
        ModelConfigInput, ModelType, ProjectedContent, Usage, OPENAI_CHAT_ENDPOINT,
        OPENAI_RESPONSES_ENDPOINT,
    },
    network::NetworkClients,
    plugin::{PluginRegistry, PluginRuntime},
    provider::{FinishReason, ModelEvent, ProviderRouter},
    store::ExternalApiSettings,
};
use serde_json::{json, Value};
use tower::ServiceExt;

fn model_input() -> ModelConfigInput {
    ModelConfigInput {
        enabled: true,
        sort_order: 0,
        display_name: "Example".into(),
        group_name: Some("work".into()),
        model_type: ModelType::OpenAi,
        base_url: "https://example.com/v1".into(),
        use_full_url: false,
        api_key: "upstream".into(),
        tooltip_data: "test".into(),
        model_id: "qwen/model".into(),
        reasoning_effort: None,
        openai_endpoint: OPENAI_CHAT_ENDPOINT.into(),
        openai_extra_params_enabled: false,
        openai_extra_params: json!({}),
        custom_headers_enabled: false,
        custom_headers: json!({}),
        anthropic_extra_params_enabled: false,
        anthropic_extra_params: json!({}),
        context_window_tokens: None,
        max_completion_tokens: None,
        anthropic_max_tokens: None,
        anthropic_thinking_effort: None,
        thinking_budget_tokens: None,
    }
}

async fn setup() -> (
    axum::Router,
    fake_provider::FakeProvider,
    cursor_server::store::Store,
    tempfile::TempDir,
) {
    let (directory, store) = fixtures::temp_store().await;
    store.create_model(&model_input()).await.unwrap();
    let runtime = PluginRuntime::managed().unwrap();
    let plugins = PluginRegistry::managed(store.clone(), runtime.clone(), "0.1.0".into()).unwrap();
    let provider = fake_provider::FakeProvider::default();
    let shared_provider = Arc::new(provider.clone());
    let control = control::ControlService::new(
        store.clone(),
        shared_provider.clone(),
        runtime,
        plugins.clone(),
        NetworkClients::new(store.clone()),
    )
    .unwrap();
    let router = byok::router(store.clone(), plugins, shared_provider, None)
        .merge(control::api_router(control));
    (router, provider, store, directory)
}

#[tokio::test]
async fn management_settings_enable_the_external_route_without_restart() {
    let (router, _provider, _store, _directory) = setup().await;
    let (status, body) = send(
        router.clone(),
        "GET",
        "/__byok-api__/api/settings/external-api",
        None,
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["enabled"],
        false
    );
    let (status, _) = send(
        router.clone(),
        "PUT",
        "/__byok-api__/api/settings/external-api",
        None,
        json!({"enabled":true,"api_key":"changed-key"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        send(
            router,
            "GET",
            "/byok/v1/models",
            Some("changed-key"),
            json!({})
        )
        .await
        .0,
        StatusCode::OK
    );
}

async fn send(
    router: axum::Router,
    method: &str,
    path: &str,
    key: Option<&str>,
    body: Value,
) -> (StatusCode, String) {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let response = router
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

#[tokio::test]
async fn disabled_and_unauthorized_requests_cannot_list_models() {
    let (router, _provider, store, _directory) = setup().await;
    assert_eq!(
        send(
            router.clone(),
            "GET",
            "/byok/v1/models",
            Some("secret"),
            json!({})
        )
        .await
        .0,
        StatusCode::FORBIDDEN
    );
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    assert_eq!(
        send(router.clone(), "GET", "/byok/v1/models", None, json!({}))
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        send(
            router.clone(),
            "GET",
            "/byok/v1/models",
            Some("wrong"),
            json!({})
        )
        .await
        .0,
        StatusCode::UNAUTHORIZED
    );
    let (status, body) = send(router, "GET", "/byok/v1/models", Some("secret"), json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["data"][0]["id"],
        "qwen/model"
    );
}

#[tokio::test]
async fn responses_accept_the_configured_model_id_without_a_group() {
    let (router, provider, store, _directory) = setup().await;
    let mut input = model_input();
    input.group_name = None;
    input.model_id = "gpt-6.1-sol".into();
    input.openai_endpoint = OPENAI_RESPONSES_ENDPOINT.into();
    let model = store.create_model(&input).await.unwrap();
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();

    provider.push(vec![
        ModelEvent::TextDelta("ok".into()),
        ModelEvent::Done(FinishReason::Stop),
    ]);
    let (status, body) = send(
        router.clone(),
        "POST",
        "/byok/v1/responses",
        Some("secret"),
        json!({"model":"gpt-6.1-sol","input":"hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(provider.requests()[0].model.model_id, model.model_hash);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["model"],
        "gpt-6.1-sol"
    );

    let (status, body) = send(router, "GET", "/byok/v1/models", Some("secret"), json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let listed = serde_json::from_str::<Value>(&body).unwrap();
    assert!(listed["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|model| model["id"] == "gpt-6.1-sol"));
}

#[tokio::test]
async fn disabled_models_are_excluded_until_reenabled() {
    let (router, provider, store, _directory) = setup().await;
    let first = store.models().await.unwrap().remove(0);
    let mut duplicate = model_input();
    duplicate.sort_order = 1;
    duplicate.base_url = "https://backup.example.com/v1".into();
    let second = store.create_model(&duplicate).await.unwrap();
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();

    let mut update = serde_json::to_value(&first).unwrap();
    for (enabled, candidates) in [
        (false, vec![&second.model_hash, &second.model_hash]),
        (true, vec![&first.model_hash, &second.model_hash]),
    ] {
        update["enabled"] = json!(enabled);
        let (status, body) = send(
            router.clone(),
            "PUT",
            &format!("/__byok-api__/api/models/{}", first.model_hash),
            None,
            update.clone(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["enabled"],
            enabled
        );
        for expected in candidates {
            provider.push(vec![
                ModelEvent::TextDelta("ok".into()),
                ModelEvent::Done(FinishReason::Stop),
            ]);
            let (status, body) = send(
                router.clone(),
                "POST",
                "/byok/v1/responses",
                Some("secret"),
                json!({"model":first.model_id,"input":"hello"}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(
                &provider.requests().last().unwrap().model.model_id,
                expected
            );
        }
    }

    for model in [&first, &second] {
        let mut update = serde_json::to_value(model).unwrap();
        update["enabled"] = json!(false);
        assert_eq!(
            send(
                router.clone(),
                "PUT",
                &format!("/__byok-api__/api/models/{}", model.model_hash),
                None,
                update
            )
            .await
            .0,
            StatusCode::OK
        );
    }
    let (_, body) = send(
        router.clone(),
        "GET",
        "/byok/v1/models",
        Some("secret"),
        json!({}),
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["data"],
        json!([])
    );
    for (path, body) in [
        (
            "/byok/v1/responses",
            json!({"model":first.model_id,"input":"hello"}),
        ),
        (
            "/byok/v1/chat/completions",
            json!({"model":first.model_id,"messages":[]}),
        ),
        (
            "/byok/v1/messages",
            json!({"model":first.model_id,"messages":[],"max_tokens":32}),
        ),
    ] {
        assert_eq!(
            send(router.clone(), "POST", path, Some("secret"), body)
                .await
                .0,
            StatusCode::NOT_FOUND
        );
    }
    assert_eq!(store.models().await.unwrap().len(), 2);
    assert_eq!(provider.requests().len(), 4);
    provider.push(vec![
        ModelEvent::TextDelta("1 2 3".into()),
        ModelEvent::Done(FinishReason::Stop),
    ]);
    let (status, body) = send(
        router,
        "POST",
        &format!(
            "/__byok-api__/api/models/{}/test/disabled-test",
            first.model_hash
        ),
        None,
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        !store
            .model(&first.model_hash)
            .await
            .unwrap()
            .unwrap()
            .enabled
    );
}

#[tokio::test]
async fn duplicate_model_ids_rotate_in_configuration_order() {
    let (router, provider, store, _directory) = setup().await;
    let first = store.models().await.unwrap().remove(0);
    let mut duplicate = model_input();
    duplicate.sort_order = 1;
    duplicate.display_name = "Backup".into();
    duplicate.group_name = Some("backup".into());
    duplicate.base_url = "https://backup.example.com/v1".into();
    let second = store.create_model(&duplicate).await.unwrap();
    let mut distinct = model_input();
    distinct.sort_order = 2;
    distinct.model_id = "qwen/other".into();
    let distinct = store.create_model(&distinct).await.unwrap();
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();

    let (status, body) = send(
        router.clone(),
        "GET",
        "/byok/v1/models",
        Some("secret"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let models = serde_json::from_str::<Value>(&body).unwrap();
    let ids = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids, ["qwen/model", "qwen/other"]);

    let first = first.model_hash;
    let second = second.model_hash;
    for (public_id, expected) in [
        ("qwen/model", &first),
        ("qwen/other", &distinct.model_hash),
        ("qwen/model", &second),
        ("qwen/model", &first),
        ("qwen/model", &second),
    ] {
        assert_eq!(
            send(
                router.clone(),
                "GET",
                "/byok/v1/models",
                Some("secret"),
                json!({})
            )
            .await
            .0,
            StatusCode::OK
        );
        assert_eq!(
            send(
                router.clone(),
                "POST",
                "/byok/v1/responses",
                Some("wrong"),
                json!({"model":public_id,"input":"hello"})
            )
            .await
            .0,
            StatusCode::UNAUTHORIZED
        );
        provider.push(vec![
            ModelEvent::TextDelta("ok".into()),
            ModelEvent::Done(FinishReason::Stop),
        ]);
        let (status, body) = send(
            router.clone(),
            "POST",
            "/byok/v1/responses",
            Some("secret"),
            json!({"model":public_id,"input":"hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            &provider.requests().last().unwrap().model.model_id,
            expected
        );
    }
}

#[tokio::test]
async fn concurrent_requests_share_the_model_rotation() {
    let (router, provider, store, _directory) = setup().await;
    let first = store.models().await.unwrap().remove(0);
    let mut duplicate = model_input();
    duplicate.sort_order = 1;
    duplicate.base_url = "https://backup.example.com/v1".into();
    let second = store.create_model(&duplicate).await.unwrap();
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();

    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..10 {
        provider.push(vec![
            ModelEvent::TextDelta("ok".into()),
            ModelEvent::Done(FinishReason::Stop),
        ]);
        let router = router.clone();
        tasks.spawn(async move {
            send(
                router,
                "POST",
                "/byok/v1/responses",
                Some("secret"),
                json!({"model":"qwen/model","input":"hello"}),
            )
            .await
        });
    }
    while let Some(result) = tasks.join_next().await {
        let (status, body) = result.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let requests = provider.requests();
    for hash in [&first.model_hash, &second.model_hash] {
        assert_eq!(
            requests
                .iter()
                .filter(|request| &request.model.model_id == hash)
                .count(),
            5
        );
    }
}

#[tokio::test]
async fn all_three_protocols_use_the_public_model_id() {
    let (router, provider, store, _directory) = setup().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    for (path, body) in [
        (
            "/byok/v1/chat/completions",
            json!({"model":"qwen/model","messages":[{"role":"user","content":"hello"}]}),
        ),
        (
            "/byok/v1/responses",
            json!({"model":"qwen/model","input":"hello"}),
        ),
        (
            "/byok/v1/messages",
            json!({"model":"qwen/model","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}),
        ),
    ] {
        provider.push(vec![
            ModelEvent::TextStart,
            ModelEvent::TextDelta("world".into()),
            ModelEvent::TextEnd,
            ModelEvent::Done(FinishReason::Stop),
        ]);
        let (status, response) = send(router.clone(), "POST", path, Some("secret"), body).await;
        assert_eq!(status, StatusCode::OK, "{path}: {response}");
        assert!(response.contains("world"), "{path}: {response}");
    }
    assert_eq!(provider.requests().len(), 3);
    assert!(provider
        .requests()
        .iter()
        .all(|request| request.model.model_id != "qwen/model"));
}

#[tokio::test]
async fn streaming_chat_returns_incremental_sse_and_tool_calls() {
    let (router, provider, store, _directory) = setup().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    provider.push(vec![
        ModelEvent::TextStart,
        ModelEvent::TextDelta("hello".into()),
        ModelEvent::TextEnd,
        ModelEvent::ToolCallStart {
            index: 0,
            call_id: "call_1".into(),
            name: "lookup".into(),
        },
        ModelEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "{\"q\":1}".into(),
        },
        ModelEvent::ToolCallEnd { index: 0 },
        ModelEvent::Done(FinishReason::ToolUse),
    ]);
    let (status, body) = send(
        router,
        "POST",
        "/byok/v1/chat/completions",
        Some("secret"),
        json!({"model":"qwen/model","stream":true,"messages":[{"role":"user","content":"hello"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("lookup"));
    assert!(body.contains("tool_calls"));
    assert!(body.contains("[DONE]"));
}

#[tokio::test]
async fn chat_tool_result_keeps_the_assistant_function_name() {
    let (router, provider, store, _directory) = setup().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    provider.push(vec![
        ModelEvent::TextDelta("done".into()),
        ModelEvent::Done(FinishReason::Stop),
    ]);
    let body = json!({"model":"qwen/model","messages":[
        {"role":"user","content":"find it"},
        {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}]},
        {"role":"tool","tool_call_id":"call_1","content":"found"}
    ]});
    assert_eq!(
        send(
            router,
            "POST",
            "/byok/v1/chat/completions",
            Some("secret"),
            body
        )
        .await
        .0,
        StatusCode::OK
    );
    let requests = provider.requests();
    let ProjectedContent::ToolResult(result) = &requests[0].history[2].content else {
        panic!("expected tool result");
    };
    assert_eq!(result.name, "lookup");
}

#[tokio::test]
async fn responses_and_messages_stream_with_protocol_end_events() {
    let (router, provider, store, _directory) = setup().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    for (path, request, terminal) in [
        (
            "/byok/v1/responses",
            json!({"model":"qwen/model","input":"hello","stream":true}),
            "response.completed",
        ),
        (
            "/byok/v1/messages",
            json!({"model":"qwen/model","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"stream":true}),
            "message_stop",
        ),
    ] {
        provider.push(vec![
            ModelEvent::TextDelta("world".into()),
            ModelEvent::Done(FinishReason::Stop),
        ]);
        let (status, body) = send(router.clone(), "POST", path, Some("secret"), request).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains(terminal), "{path}: {body}");
    }
}

#[tokio::test]
async fn responses_stream_emits_complete_text_and_tool_item_lifecycles() {
    let (router, provider, store, _directory) = setup().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    provider.push(vec![
        ModelEvent::TextStart,
        ModelEvent::TextDelta("hello".into()),
        ModelEvent::TextEnd,
        ModelEvent::ToolCallStart {
            index: 0,
            call_id: "call_1".into(),
            name: "lookup".into(),
        },
        ModelEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "{\"q\":1}".into(),
        },
        ModelEvent::ToolCallEnd { index: 0 },
        ModelEvent::Done(FinishReason::ToolUse),
    ]);
    let (status, body) = send(
        router,
        "POST",
        "/byok/v1/responses",
        Some("secret"),
        json!({"model":"qwen/model","input":"hello","stream":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = body
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect::<Vec<_>>();
    assert_eq!(
        events,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert!(body.contains("\"output_index\":1"), "{body}");
}

#[tokio::test]
async fn messages_stream_closes_each_content_block_before_stopping() {
    let (router, provider, store, _directory) = setup().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    provider.push(vec![
        ModelEvent::TextStart,
        ModelEvent::TextDelta("hello".into()),
        ModelEvent::TextEnd,
        ModelEvent::ToolCallStart {
            index: 0,
            call_id: "call_1".into(),
            name: "lookup".into(),
        },
        ModelEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "{\"q\":1}".into(),
        },
        ModelEvent::ToolCallEnd { index: 0 },
        ModelEvent::Done(FinishReason::ToolUse),
    ]);
    let (status, body) = send(router, "POST", "/byok/v1/messages", Some("secret"),
        json!({"model":"qwen/model","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"stream":true})).await;
    assert_eq!(status, StatusCode::OK);
    let events = body
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect::<Vec<_>>();
    assert_eq!(
        events,
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert!(body.contains("\"index\":1"), "{body}");
}

#[tokio::test]
async fn all_protocols_preserve_cached_usage_in_streaming_and_complete_responses() {
    let (router, provider, store, _directory) = setup().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    let usage = Usage {
        input_tokens: Some(1_000),
        context_input_tokens: Some(1_000),
        output_tokens: Some(50),
        total_tokens: Some(1_050),
        cache_read_tokens: Some(800),
        cache_write_tokens: Some(20),
        reasoning_tokens: Some(10),
    };
    for (path, request, usage_pointer, cached_pointer, expected_input) in [
        (
            "/byok/v1/chat/completions",
            json!({"model":"qwen/model","messages":[{"role":"user","content":"hello"}]}),
            "/usage",
            "/prompt_tokens_details/cached_tokens",
            1_000,
        ),
        (
            "/byok/v1/responses",
            json!({"model":"qwen/model","input":"hello"}),
            "/response/usage",
            "/input_tokens_details/cached_tokens",
            1_000,
        ),
        (
            "/byok/v1/messages",
            json!({"model":"qwen/model","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}),
            "/usage",
            "/cache_read_input_tokens",
            180,
        ),
    ] {
        for stream in [false, true] {
            provider.push(vec![
                ModelEvent::TextDelta("world".into()),
                ModelEvent::Usage(usage),
                ModelEvent::Done(FinishReason::Stop),
            ]);
            let mut request = request.clone();
            request["stream"] = json!(stream);
            let (status, body) = send(router.clone(), "POST", path, Some("secret"), request).await;
            assert_eq!(status, StatusCode::OK, "{path}: {body}");
            let response = if stream {
                body.lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .find(|event| match path {
                        "/byok/v1/chat/completions" => event.get("usage").is_some(),
                        "/byok/v1/responses" => event["type"] == "response.completed",
                        _ => event["type"] == "message_delta",
                    })
                    .unwrap_or_else(|| panic!("missing usage event in {path}: {body}"))
            } else {
                serde_json::from_str::<Value>(&body).unwrap()
            };
            let usage = response
                .pointer(if stream { usage_pointer } else { "/usage" })
                .unwrap();
            assert_eq!(
                usage.pointer(cached_pointer),
                Some(&json!(800)),
                "{path} stream={stream}: {body}"
            );
            let input_field = if path == "/byok/v1/chat/completions" {
                "prompt_tokens"
            } else {
                "input_tokens"
            };
            assert_eq!(
                usage[input_field], expected_input,
                "{path} stream={stream}: {body}"
            );
            if path == "/byok/v1/messages" {
                assert_eq!(usage["cache_creation_input_tokens"], 20, "{body}");
            }
            if stream && path == "/byok/v1/chat/completions" {
                let finished = body.find("\"finish_reason\":\"stop\"").unwrap();
                let usage_position = body.find("\"cached_tokens\":800").unwrap();
                let done = body.find("[DONE]").unwrap();
                assert!(finished < usage_position && usage_position < done, "{body}");
            }
        }
    }
}

#[tokio::test]
async fn all_entry_and_upstream_protocol_pairs_preserve_cache_usage() {
    let (_directory, store) = fixtures::temp_store().await;
    let (_inner_directory, inner_store) = fixtures::temp_store().await;
    for group in ["bridge-chat", "bridge-responses", "bridge-messages"] {
        let mut model = model_input();
        model.model_id = format!("{group}/qwen/model");
        inner_store.create_model(&model).await.unwrap();
    }
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    inner_store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    let runtime = PluginRuntime::managed().unwrap();
    let inner_plugins =
        PluginRegistry::managed(inner_store.clone(), runtime.clone(), "0.1.0".into()).unwrap();
    let plugins = PluginRegistry::managed(store.clone(), runtime, "0.1.0".into()).unwrap();
    let fake = fake_provider::FakeProvider::default();
    let inner = byok::router(inner_store, inner_plugins, Arc::new(fake.clone()), None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move { axum::serve(listener, inner).await.unwrap() });

    for (order, group, model_type, endpoint) in [
        (1, "bridge-chat", ModelType::OpenAi, OPENAI_CHAT_ENDPOINT),
        (
            2,
            "bridge-responses",
            ModelType::OpenAi,
            OPENAI_RESPONSES_ENDPOINT,
        ),
        (3, "bridge-messages", ModelType::Anthropic, ""),
    ] {
        let mut model = model_input();
        model.sort_order = order;
        model.display_name = format!("Bridge {group}");
        model.group_name = Some(group.into());
        model.model_type = model_type;
        model.openai_endpoint = endpoint.into();
        model.base_url = format!("http://127.0.0.1:{port}/byok/v1");
        model.model_id = format!("{group}/qwen/model");
        model.api_key = "secret".into();
        store.create_model(&model).await.unwrap();
    }
    let provider = ProviderRouter::new(
        store.clone(),
        plugins.clone(),
        NetworkClients::new(store.clone()),
        Duration::from_secs(10),
        Duration::from_secs(10),
    );
    let outer = byok::router(
        store.clone(),
        plugins,
        Arc::new(provider),
        Some(byok::NativeForwarder::new(
            store.clone(),
            NetworkClients::new(store.clone()),
            Duration::from_secs(10),
            Duration::from_secs(10),
        )),
    );
    for (path, request) in [
        (
            "/byok/v1/chat/completions",
            json!({"messages":[{"role":"user","content":"hello"}],
                "tools":[{"type":"function","function":{"name":"lookup","description":"Look up a value","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]}),
        ),
        (
            "/byok/v1/responses",
            json!({"input":"hello",
            "tools":[{"type":"function","name":"lookup","description":"Look up a value","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]}),
        ),
        (
            "/byok/v1/messages",
            json!({"max_tokens":100,"messages":[{"role":"user","content":"hello"}],
                "tools":[{"name":"lookup","description":"Look up a value","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}]}),
        ),
    ] {
        for group in ["bridge-chat", "bridge-responses", "bridge-messages"] {
            for stream in [false, true] {
                let prior_calls: HashSet<_> = store
                    .llm_calls(100)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|call| call.call_id)
                    .collect();
                fake.push(vec![
                    ModelEvent::TextStart,
                    ModelEvent::TextDelta("world".into()),
                    ModelEvent::TextEnd,
                    ModelEvent::ToolCallStart {
                        index: 0,
                        call_id: "call_1".into(),
                        name: "lookup".into(),
                    },
                    ModelEvent::ToolCallArgumentsDelta {
                        index: 0,
                        delta: "{\"q\":\"x\"}".into(),
                    },
                    ModelEvent::ToolCallEnd { index: 0 },
                    ModelEvent::Usage(Usage {
                        input_tokens: Some(1_000),
                        context_input_tokens: Some(1_000),
                        output_tokens: Some(50),
                        total_tokens: Some(1_050),
                        cache_read_tokens: Some(800),
                        ..Usage::default()
                    }),
                    ModelEvent::Done(FinishReason::ToolUse),
                ]);
                let mut request = request.clone();
                request["model"] = json!(format!("{group}/qwen/model"));
                request["stream"] = json!(stream);
                let (status, body) =
                    send(outer.clone(), "POST", path, Some("secret"), request).await;
                assert_eq!(
                    status,
                    StatusCode::OK,
                    "{path} -> {group} stream={stream}: {body}"
                );
                assert!(
                    body.contains("world"),
                    "{path} -> {group} stream={stream}: {body}"
                );
                assert!(
                    body.contains("lookup") && body.contains("call_1"),
                    "{path} -> {group} stream={stream}: {body}"
                );
                let requests = fake.requests();
                let forwarded = requests.last().unwrap();
                assert_eq!(
                    forwarded.prompt.tools.len(),
                    1,
                    "{path} -> {group} stream={stream}"
                );
                assert_eq!(
                    forwarded.prompt.tools[0].name, "lookup",
                    "{path} -> {group} stream={stream}"
                );
                let calls = store.llm_calls(100).await.unwrap();
                let call = calls
                    .iter()
                    .find(|call| {
                        call.display_name == format!("Bridge {group}")
                            && !prior_calls.contains(&call.call_id)
                    })
                    .unwrap();
                assert_eq!(
                    call.cache_read_tokens,
                    Some(800),
                    "{path} -> {group} stream={stream}"
                );
            }
        }
    }
    assert_eq!(fake.requests().len(), 18);
    server.abort();
}

#[tokio::test]
async fn native_responses_rotate_between_same_named_models() {
    let (_directory, store) = fixtures::temp_store().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let upstream = Router::new().route(
        "/{channel}/v1/responses",
        post(
            move |axum::extract::Path(channel): axum::extract::Path<String>,
                  Json(body): Json<Value>| {
                let sender = sender.clone();
                async move {
                    sender.send((channel, body)).unwrap();
                    Json(json!({"object":"response","status":"completed","output":[]}))
                }
            },
        ),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    for (sort_order, channel) in [(0, "first"), (1, "second")] {
        let mut input = model_input();
        input.sort_order = sort_order;
        input.model_id = "gpt-6.1-sol".into();
        input.openai_endpoint = OPENAI_RESPONSES_ENDPOINT.into();
        input.base_url = format!("http://127.0.0.1:{port}/{channel}/v1/responses");
        input.use_full_url = true;
        store.create_model(&input).await.unwrap();
    }
    let runtime = PluginRuntime::managed().unwrap();
    let plugins = PluginRegistry::managed(store.clone(), runtime, "0.1.0".into()).unwrap();
    let router = byok::router(
        store.clone(),
        plugins,
        Arc::new(fake_provider::FakeProvider::default()),
        Some(byok::NativeForwarder::new(
            store.clone(),
            NetworkClients::new(store),
            Duration::from_secs(10),
            Duration::from_secs(10),
        )),
    );
    for channel in ["first", "second", "first", "second"] {
        let (status, body) = send(
            router.clone(),
            "POST",
            "/byok/v1/responses",
            Some("secret"),
            json!({"model":"gpt-6.1-sol","input":"hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (selected, forwarded) = receiver.recv().await.unwrap();
        assert_eq!(selected, channel);
        assert_eq!(forwarded["model"], "gpt-6.1-sol");
    }
    server.abort();
}

#[tokio::test]
async fn matching_http_protocols_forward_native_requests_and_responses() {
    let (directory, store) = fixtures::temp_store().await;
    store
        .set_external_api_settings(ExternalApiSettings {
            enabled: true,
            api_key: "secret".into(),
        })
        .await
        .unwrap();
    let runtime = PluginRuntime::managed().unwrap();
    let plugins = PluginRegistry::managed(store.clone(), runtime, "0.1.0".into()).unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<(HeaderMap, Value)>();
    let upstream = Router::new().route(
        "/{*path}",
        post(move |headers: HeaderMap, Json(body): Json<Value>| {
            let sender = sender.clone();
            async move {
                sender.send((headers, body.clone())).unwrap();
                if body["native_error"] == true {
                    return (
                        StatusCode::TOO_MANY_REQUESTS,
                        Json(json!({"error":"native-rate-limit"})),
                    )
                        .into_response();
                }
                if body["stream"] == true {
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        "data: {\"native_marker\":\"untouched-stream\"}\n\n",
                    )
                        .into_response()
                } else {
                    Json(json!({"native_marker":"untouched-complete","echo":body})).into_response()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    for (order, group, model_type, endpoint, path, request) in [
        (
            1,
            "chat",
            ModelType::OpenAi,
            OPENAI_CHAT_ENDPOINT,
            "/byok/v1/chat/completions",
            json!({"messages":[{"role":"user","content":"hi"}],"native_extension":{"keep":1}}),
        ),
        (
            2,
            "responses",
            ModelType::OpenAi,
            OPENAI_RESPONSES_ENDPOINT,
            "/byok/v1/responses",
            json!({"input":[{"type":"native_unsupported","value":1}],"native_extension":{"keep":2}}),
        ),
        (
            3,
            "messages",
            ModelType::Anthropic,
            "",
            "/byok/v1/messages",
            json!({"max_tokens":100,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"url","url":"https://example.com"}}]}],"native_extension":{"keep":3}}),
        ),
    ] {
        let mut model = model_input();
        model.sort_order = order;
        model.group_name = Some(group.into());
        model.model_type = model_type;
        model.openai_endpoint = endpoint.into();
        model.base_url = format!("http://127.0.0.1:{port}/byok/v1");
        model.model_id = format!("{group}-native-model");
        store.create_model(&model).await.unwrap();
        for stream in [false, true] {
            let mut request = request.clone();
            request["model"] = json!(model.model_id);
            request["stream"] = json!(stream);
            let (status, body) = send(
                byok::router(
                    store.clone(),
                    plugins.clone(),
                    Arc::new(fake_provider::FakeProvider::default()),
                    Some(byok::NativeForwarder::new(
                        store.clone(),
                        NetworkClients::new(store.clone()),
                        Duration::from_secs(10),
                        Duration::from_secs(10),
                    )),
                ),
                "POST",
                path,
                Some("secret"),
                request.clone(),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{path} stream={stream}: {body}");
            if stream {
                assert_eq!(body, "data: {\"native_marker\":\"untouched-stream\"}\n\n");
            } else {
                assert_eq!(
                    serde_json::from_str::<Value>(&body).unwrap()["native_marker"],
                    "untouched-complete"
                );
            }
            let (upstream_headers, forwarded) = receiver.recv().await.unwrap();
            assert_eq!(forwarded, request);
            if model_type == ModelType::Anthropic {
                assert_eq!(upstream_headers["x-api-key"], "upstream");
                assert_eq!(upstream_headers["anthropic-version"], "2023-06-01");
                assert!(!upstream_headers.contains_key(header::AUTHORIZATION));
            } else {
                assert_eq!(upstream_headers[header::AUTHORIZATION], "Bearer upstream");
                assert!(!upstream_headers.contains_key("x-api-key"));
            }
        }
    }
    let error_request = json!({"model":"chat-native-model","messages":[],"native_error":true});
    let (status, body) = send(
        byok::router(
            store.clone(),
            plugins,
            Arc::new(fake_provider::FakeProvider::default()),
            Some(byok::NativeForwarder::new(
                store.clone(),
                NetworkClients::new(store.clone()),
                Duration::from_secs(10),
                Duration::from_secs(10),
            )),
        ),
        "POST",
        "/byok/v1/chat/completions",
        Some("secret"),
        error_request.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"],
        "native-rate-limit"
    );
    assert_eq!(receiver.recv().await.unwrap().1, error_request);
    server.abort();
    drop(directory);
}
