-- Keep the last upstream failure visible after a model is automatically disabled.
ALTER TABLE model_configs ADD COLUMN last_error TEXT;
ALTER TABLE model_configs ADD COLUMN last_error_at_ms INTEGER;
