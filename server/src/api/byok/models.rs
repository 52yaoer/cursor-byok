use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::{plugin::PluginRegistry, store::Store, Result};

#[derive(Clone)]
pub(super) struct ListedModel {
    pub public_id: String,
    pub internal_id: String,
    pub display_name: String,
}

const RETRY_DELAYS: [Duration; 5] = [
    Duration::from_secs(5),
    Duration::from_secs(15),
    Duration::from_secs(60),
    Duration::from_secs(5 * 60),
    Duration::from_secs(15 * 60),
];

#[derive(Default)]
struct FailureState {
    failures: u32,
    retry_at: Option<Instant>,
}

/// Runtime health is separate from the user's persistent enabled switch.
#[derive(Clone, Default)]
pub(super) struct ModelHealth {
    failures: Arc<Mutex<HashMap<String, FailureState>>>,
}

impl ModelHealth {
    pub fn is_available(&self, model_id: &str) -> bool {
        let mut failures = self.failures.lock();
        let Some(state) = failures.get_mut(model_id) else {
            return true;
        };
        if state
            .retry_at
            .is_some_and(|retry_at| retry_at <= Instant::now())
        {
            state.retry_at = None;
            return true;
        }
        state.retry_at.is_none()
    }

    pub fn record_failure(&self, model_id: &str, _error: impl Into<String>) {
        let mut failures = self.failures.lock();
        let state = failures.entry(model_id.to_owned()).or_default();
        state.failures = state.failures.saturating_add(1);
        let index = state.failures.saturating_sub(1) as usize;
        let delay = RETRY_DELAYS[index.min(RETRY_DELAYS.len() - 1)];
        state.retry_at = Some(Instant::now() + delay);
        tracing::warn!(
            model_id,
            failures = state.failures,
            ?delay,
            "model temporarily cooled down after upstream failure"
        );
    }

    pub fn record_success(&self, model_id: &str) {
        self.failures.lock().remove(model_id);
    }
}

#[derive(Default)]
pub(super) struct ModelRotation {
    next: Mutex<HashMap<String, usize>>,
    health: ModelHealth,
}

impl ModelRotation {
    pub fn candidates<'a>(
        &self,
        models: &'a [ListedModel],
        public_id: &str,
    ) -> Vec<&'a ListedModel> {
        let candidates = models
            .iter()
            .filter(|model| model.public_id == public_id)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut next = self.next.lock();
        let next = next.entry(public_id.to_owned()).or_default();
        let index = *next % candidates.len();
        *next = (index + 1) % candidates.len();
        candidates
            .iter()
            .cycle()
            .skip(index)
            .take(candidates.len())
            .copied()
            .filter(|model| self.health.is_available(&model.internal_id))
            .collect()
    }

    pub fn health(&self) -> ModelHealth {
        self.health.clone()
    }
}

pub(super) async fn list(store: &Store, plugins: &PluginRegistry) -> Result<Vec<ListedModel>> {
    let mut models = Vec::new();
    let mut configured = store.enabled_models().await?;
    configured.sort_by(|left, right| {
        left.sort_order
            .cmp(&right.sort_order)
            .then_with(|| left.display_name.cmp(&right.display_name))
            .then_with(|| left.model_hash.cmp(&right.model_hash))
    });
    for model in configured {
        models.push(ListedModel {
            public_id: model.model_id,
            internal_id: model.model_hash,
            display_name: model.display_name,
        });
    }
    for model in plugins.configured_models().await {
        let public_id = format!(
            "plugin:{}:{}/{}",
            model.plugin_id, model.provider_id, model.model_id
        );
        models.push(ListedModel {
            public_id,
            internal_id: model.id,
            display_name: model.display_name,
        });
    }
    Ok(models)
}

pub(super) fn response(models: &[ListedModel]) -> Value {
    let mut ids = HashSet::new();
    json!({"object":"list","data":models.iter().filter(|model| ids.insert(&model.public_id)).map(|model| json!({
        "id":model.public_id,"object":"model","created":0,"owned_by":"cursor-byok",
        "name":model.display_name
    })).collect::<Vec<_>>()})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_failure_cools_model_without_changing_user_availability() {
        let health = ModelHealth::default();
        assert!(health.is_available("model-a"));

        health.record_failure("model-a", "upstream returned 502");
        assert!(!health.is_available("model-a"));

        health.record_success("model-a");
        assert!(health.is_available("model-a"));
    }

    #[test]
    fn rotation_skips_a_cooled_channel_and_keeps_its_user_switch_untouched() {
        let rotation = ModelRotation::default();
        let models = vec![
            ListedModel {
                public_id: "gpt".into(),
                internal_id: "channel-a".into(),
                display_name: "A".into(),
            },
            ListedModel {
                public_id: "gpt".into(),
                internal_id: "channel-b".into(),
                display_name: "B".into(),
            },
        ];

        assert_eq!(rotation.candidates(&models, "gpt").len(), 2);
        rotation.health.record_failure("channel-a", "upstream 502");
        let candidates = rotation.candidates(&models, "gpt");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].internal_id, "channel-b");
    }
}
