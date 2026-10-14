use std::collections::{HashMap, HashSet};

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::{plugin::PluginRegistry, store::Store, Result};

#[derive(Clone)]
pub(super) struct ListedModel {
    pub public_id: String,
    pub internal_id: String,
    pub display_name: String,
}

#[derive(Default)]
pub(super) struct ModelRotation {
    next: Mutex<HashMap<String, usize>>,
}

impl ModelRotation {
    pub fn select<'a>(
        &self,
        models: &'a [ListedModel],
        public_id: &str,
    ) -> Option<&'a ListedModel> {
        let candidates = models
            .iter()
            .filter(|model| model.public_id == public_id)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        let mut next = self.next.lock();
        let next = next.entry(public_id.to_owned()).or_default();
        let index = *next % candidates.len();
        *next = (index + 1) % candidates.len();
        Some(candidates[index])
    }
}

pub(super) async fn list(store: &Store, plugins: &PluginRegistry) -> Result<Vec<ListedModel>> {
    let mut models = Vec::new();
    let mut configured = store.models().await?;
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
