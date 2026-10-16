//! External OpenAI and Anthropic compatible API.
mod direct;
mod models;
mod output;
mod protocol;

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::{
    model::ModelInvocation, plugin::PluginRegistry, provider::Provider, store::Store, Error,
};

use self::protocol::Protocol;
pub use direct::NativeForwarder;

#[derive(Clone)]
struct ApiState {
    store: Store,
    plugins: PluginRegistry,
    provider: Arc<dyn Provider>,
    native: Option<NativeForwarder>,
    rotation: Arc<models::ModelRotation>,
}

pub fn router(
    store: Store,
    plugins: PluginRegistry,
    provider: Arc<dyn Provider>,
    native: Option<NativeForwarder>,
) -> Router {
    Router::new()
        .route("/byok/v1/models", get(list_models))
        .route("/byok/v1/chat/completions", post(chat))
        .route("/byok/v1/responses", post(responses))
        .route("/byok/v1/messages", post(messages))
        .with_state(ApiState {
            store,
            plugins,
            provider,
            native,
            rotation: Arc::default(),
        })
}

fn api_error(status: StatusCode, message: impl std::fmt::Display) -> Response {
    (status, Json(json!({"error":{"message":message.to_string(),"type":if status.is_client_error() { "invalid_request_error" } else { "server_error" }}}))).into_response()
}

async fn authorized(
    state: &ApiState,
    headers: &HeaderMap,
) -> std::result::Result<(), Box<Response>> {
    let settings = state
        .store
        .external_api_settings()
        .await
        .map_err(|error| Box::new(api_error(StatusCode::INTERNAL_SERVER_ERROR, error)))?;
    if !settings.enabled {
        return Err(Box::new(api_error(
            StatusCode::FORBIDDEN,
            "external API is disabled",
        )));
    }
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        })
        .unwrap_or_default();
    if supplied.is_empty() || !constant_time_eq(supplied.as_bytes(), settings.api_key.as_bytes()) {
        return Err(Box::new(api_error(
            StatusCode::UNAUTHORIZED,
            "invalid API key",
        )));
    }
    Ok(())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let difference = left
        .iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        });
    difference == 0
}

async fn list_models(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorized(&state, &headers).await {
        return *error;
    }
    match models::list(&state.store, &state.plugins).await {
        Ok(models) => Json(models::response(&models)).into_response(),
        Err(error) => api_error(StatusCode::CONFLICT, error),
    }
}

async fn chat(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    generate(state, headers, body, Protocol::Chat).await
}

async fn responses(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    generate(state, headers, body, Protocol::Responses).await
}

async fn messages(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    generate(state, headers, body, Protocol::Messages).await
}

async fn generate(
    state: ApiState,
    headers: HeaderMap,
    body: Value,
    protocol: Protocol,
) -> Response {
    if let Err(error) = authorized(&state, &headers).await {
        return *error;
    }
    let Some(public_model_id) = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    else {
        return api_error(StatusCode::BAD_REQUEST, "model is required");
    };
    let models = match models::list(&state.store, &state.plugins).await {
        Ok(models) => models,
        Err(error) => return api_error(StatusCode::CONFLICT, error),
    };
    let configured = models
        .iter()
        .any(|model| model.public_id == public_model_id);
    let candidates = state.rotation.candidates(&models, public_model_id);
    if candidates.is_empty() {
        return if configured {
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "all configured channels for this model are temporarily cooling down",
            )
        } else {
            api_error(StatusCode::NOT_FOUND, "model not found")
        };
    }
    let health = state.rotation.health();
    let mut last_error = None;
    let mut last_transient_response = None;
    let mut native_attempted = false;
    if let Some(native) = &state.native {
        for model in &candidates {
            match state.store.model(&model.internal_id).await {
                Ok(Some(config)) if protocol.provider_type() == config.provider_type() => {
                    native_attempted = true;
                    match native
                        .forward(protocol, &headers, body.clone(), &config, health.clone())
                        .await
                    {
                        Ok(response) if response.status().is_success() => {
                            health.record_success(&model.internal_id);
                            return response;
                        }
                        Ok(response) if is_transient_status(response.status()) => {
                            health.record_failure(
                                &model.internal_id,
                                format!("upstream returned {}", response.status()),
                            );
                            last_transient_response = Some(response);
                            continue;
                        }
                        Ok(response) => {
                            if is_permanent_status(response.status()) {
                                disable_failed_model_or_plugin(
                                    &state.store,
                                    &state.plugins,
                                    &model.internal_id,
                                    format!("upstream returned {}", response.status()),
                                )
                                .await;
                            }
                            return response;
                        }
                        Err(error) => {
                            handle_model_failure(
                                &state.store,
                                &state.plugins,
                                &health,
                                &model.internal_id,
                                &error,
                            )
                            .await;
                            last_error = Some(error);
                        }
                    }
                }
                Ok(_) => {}
                Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
        }
        if native_attempted {
            if let Some(response) = last_transient_response {
                return response;
            }
            let error = last_error
                .take()
                .unwrap_or_else(|| Error::Provider("all native model channels failed".into()));
            return api_error(StatusCode::BAD_GATEWAY, error);
        }
    }
    let parsed = match protocol::parse(protocol, &body) {
        Ok(parsed) => parsed,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error),
    };
    let id = format!("external-api:{}", uuid::Uuid::new_v4());
    if parsed.stream {
        let Some(model) = candidates.first() else {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no healthy model channel available",
            );
        };
        let mut request = parsed.request;
        request.model.model_id = model.internal_id.clone();
        let invocation = ModelInvocation {
            call_id: id.clone(),
            run_id: id.clone(),
            conversation_id: id.clone(),
            provider_call_index: 0,
            request,
        };
        let cancellation = CancellationToken::new();
        let stream = state.provider.stream(invocation, cancellation.clone());
        output::streamed(
            protocol,
            stream,
            cancellation,
            id,
            parsed.public_model_id,
            state.store,
            state.plugins,
            model.internal_id.clone(),
            health,
        )
    } else {
        for (attempt, model) in candidates.into_iter().enumerate() {
            let mut request = parsed.request.clone();
            request.model.model_id = model.internal_id.clone();
            let attempt_id = format!("{id}:attempt-{attempt}");
            let invocation = ModelInvocation {
                call_id: attempt_id,
                run_id: id.clone(),
                conversation_id: id.clone(),
                provider_call_index: attempt as u64,
                request,
            };
            let cancellation = CancellationToken::new();
            let stream = state.provider.stream(invocation, cancellation.clone());
            match output::complete(protocol, stream, &id, &parsed.public_model_id).await {
                Ok(response) => {
                    health.record_success(&model.internal_id);
                    return response.into_response();
                }
                Err(Error::Protocol(message)) => {
                    return api_error(StatusCode::BAD_REQUEST, message)
                }
                Err(error) => {
                    handle_model_failure(
                        &state.store,
                        &state.plugins,
                        &health,
                        &model.internal_id,
                        &error,
                    )
                    .await;
                    last_error = Some(error);
                }
            }
        }
        api_error(
            StatusCode::BAD_GATEWAY,
            last_error
                .unwrap_or_else(|| Error::Provider("no healthy model channel available".into())),
        )
    }
}

fn is_transient_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn is_permanent_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::BAD_REQUEST
            | StatusCode::UNAUTHORIZED
            | StatusCode::FORBIDDEN
            | StatusCode::NOT_FOUND
            | StatusCode::UNPROCESSABLE_ENTITY
    )
}

fn is_permanent_failure(error: &Error) -> bool {
    match error {
        Error::Config(_) => true,
        Error::Provider(message) => {
            let lower = message.to_ascii_lowercase();
            [
                " 400 ",
                " 401 ",
                " 403 ",
                " 404 ",
                " 422 ",
                "invalid api key",
                "model not found",
            ]
            .iter()
            .any(|marker| lower.contains(marker))
        }
        Error::Http(_) => false,
        _ => false,
    }
}

async fn handle_model_failure(
    store: &Store,
    plugins: &PluginRegistry,
    health: &models::ModelHealth,
    model_id: &str,
    error: &Error,
) {
    let message = error.to_string();
    if is_permanent_failure(error) {
        disable_failed_model_or_plugin(store, plugins, model_id, &message).await;
    } else {
        health.record_failure(model_id, message);
    }
}

pub(super) async fn disable_failed_model(store: &Store, model_hash: &str, error: impl AsRef<str>) {
    if let Err(disable_error) = store
        .disable_model_with_error(model_hash, error.as_ref())
        .await
    {
        tracing::warn!(model_hash, %disable_error, "failed to disable unavailable model");
    }
}

pub(super) async fn disable_failed_model_or_plugin(
    store: &Store,
    plugins: &PluginRegistry,
    model_id: &str,
    error: impl AsRef<str>,
) {
    let message = error.as_ref();
    if crate::plugin::parse_model_id(model_id).is_some() {
        if let Err(disable_error) = plugins.disable_model_with_error(model_id, message).await {
            tracing::warn!(model_id, %disable_error, "failed to disable unavailable plugin model");
        }
    } else {
        disable_failed_model(store, model_id, message).await;
    }
}
