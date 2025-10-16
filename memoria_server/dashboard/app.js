const { useState, useEffect, useCallback, useMemo, useRef } = React;

function ToastContainer({ toasts, onDismiss }) {
  if (!toasts.length) {
    return null;
  }

  return (
    <div className="toast-container" role="region" aria-live="polite">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`toast ${toast.type}`}
          role="alert"
          aria-live={toast.type === "error" ? "assertive" : "polite"}
        >
          <div className="toast-content">
            <strong className="toast-title">
              {toast.type === "error" ? "Error" : "Success"}
            </strong>
            <span className="toast-message">{toast.message}</span>
          </div>
          <button
            type="button"
            className="toast-dismiss"
            onClick={() => onDismiss(toast.id)}
            aria-label="Dismiss notification"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

function buildPatch(path, value) {
  if (!Array.isArray(path) || !path.length) {
    return {};
  }
  const key = path.map((segment) => String(segment)).join(".");
  return { [key]: value };
}

function updateNestedValue(source, path, value) {
  if (!Array.isArray(path) || !path.length) {
    return source;
  }
  const [head, ...rest] = path;
  const next = Array.isArray(rest) && rest.length
    ? updateNestedValue(
        source && typeof source === "object" && source !== null
          ? source[head]
          : undefined,
        rest,
        value,
      )
    : value;
  const clone = Array.isArray(source) ? [...source] : { ...(source || {}) };
  clone[head] = next;
  return clone;
}

function getNestedValue(source, path) {
  if (!Array.isArray(path)) {
    return undefined;
  }
  return path.reduce((current, segment) => {
    if (current && typeof current === "object") {
      return current[segment];
    }
    return undefined;
  }, source);
}

function formatLabel(key) {
  if (!key) {
    return "";
  }
  const label = key.replace(/[_.-]+/g, " ").trim();
  return label.charAt(0).toUpperCase() + label.slice(1);
}

const INGESTION_TOGGLE_PATHS = [
  {
    key: "agents.conscious_ingest",
    path: ["agents", "conscious_ingest"],
    label: "Conscious ingest",
    description:
      "Enable the working-memory pipeline for personal and session-critical memories.",
  },
  {
    key: "memory.context_injection",
    path: ["memory", "context_injection"],
    label: "Context injection",
    description:
      "Stream rich, per-query context into conversations using the auto-ingest pipeline.",
  },
  {
    key: "memory.sovereign_ingest",
    path: ["memory", "sovereign_ingest"],
    label: "Sovereign/manual logging",
    description:
      "Keep ingestion manual—ideal for the lightweight direct-to-long-term capture flow.",
  },
];

function buildCategories(schema) {
  if (!schema || schema.type !== "object") {
    return [];
  }
  const sections = [];
  const properties = schema.properties || {};
  const generalProperties = {};

  Object.entries(properties).forEach(([key, descriptor]) => {
    if (
      descriptor &&
      descriptor.type === "object" &&
      descriptor.properties &&
      typeof descriptor.properties === "object"
    ) {
      sections.push({
        key,
        title: descriptor.title || formatLabel(key),
        description: descriptor.description || "",
        schema: descriptor,
      });
    } else {
      generalProperties[key] = descriptor;
    }
  });

  if (Object.keys(generalProperties).length) {
    sections.unshift({
      key: "__general__",
      title: "General",
      description: schema.description || "",
      schema: {
        type: "object",
        properties: generalProperties,
      },
    });
  }

  return sections;
}

function extractEnumOptions(fieldSchema) {
  if (!fieldSchema) {
    return null;
  }
  const metadata =
    (fieldSchema["x-memoria"] && fieldSchema["x-memoria"].enum) || null;
  const values = metadata?.values || fieldSchema.enum;
  if (!Array.isArray(values) || !values.length) {
    return null;
  }
  const labels = metadata?.labels || [];
  return values.map((value, index) => ({
    value,
    label: labels[index] || formatLabel(String(value)),
    key: String(value),
  }));
}

function isSecretField(fieldSchema) {
  return Boolean(
    fieldSchema &&
      fieldSchema["x-memoria"] &&
      fieldSchema["x-memoria"].secret,
  );
}

function isNullableField(fieldSchema) {
  return Boolean(fieldSchema && fieldSchema.nullable);
}



function createDefaultMeta() {
  return {
    database: {
      summary: "Database info unavailable (sign in to view details)",
      label: "Database",
      type: "",
      display_url: "",
      configured: false,
      masked_connection: "",
    },
    sync: {
      enabled: false,
      backend: "none",
      connection: "",
      has_connection: false,
      options_configured: false,
      channel: "",
    },
    migrations: {
      last_run: null,
      results: [],
      skipped: false,
      error: null,
      message: "",
    },
    capabilities: [],
  };
}

function IngestionModeSelector({
  settings,
  dirtySettings,
  onToggle,
  onSave,
  busy,
}) {
  const toggleState = useMemo(
    () =>
      INGESTION_TOGGLE_PATHS.map((toggle) => ({
        ...toggle,
        value: Boolean(getNestedValue(settings, toggle.path)),
      })),
    [settings],
  );

  const consciousEnabled = toggleState.find(
    (toggle) => toggle.key === "agents.conscious_ingest",
  )?.value;
  const contextEnabled = toggleState.find(
    (toggle) => toggle.key === "memory.context_injection",
  )?.value;
  const sovereignEnabled = toggleState.find(
    (toggle) => toggle.key === "memory.sovereign_ingest",
  )?.value;

  const dirtyKeys = dirtySettings || {};
  const hasPending = useMemo(
    () =>
      INGESTION_TOGGLE_PATHS.some((toggle) =>
        Object.prototype.hasOwnProperty.call(dirtyKeys, toggle.key),
      ),
    [dirtyKeys],
  );

  const summary = useMemo(() => {
    if (contextEnabled && consciousEnabled) {
      return "Combined mode: personal conscious ingest plus automated context injection.";
    }
    if (contextEnabled) {
      return "Auto-ingest mode: dynamic context injection is active for conversations.";
    }
    if (consciousEnabled) {
      return "Conscious mode: personal and session-critical memories stay in working memory.";
    }
    if (sovereignEnabled) {
      return "Manual capture: rely on the simplified direct logging flow for long-term storage.";
    }
    return "All ingestion pipelines are paused. Enable one or more to activate memory flows.";
  }, [consciousEnabled, contextEnabled, sovereignEnabled]);

  return (
    <section
      className="ingestion-mode-card"
      aria-labelledby="ingestion-mode-title"
      aria-describedby="ingestion-mode-summary"
    >
      <div className="ingestion-mode-header">
        <div>
          <h4 id="ingestion-mode-title">Ingestion pipelines</h4>
          <p className="muted" id="ingestion-mode-summary">
            {summary}
          </p>
        </div>
      </div>
      <div className="ingestion-toggle-list">
        {toggleState.map((toggle) => (
          <label key={toggle.key} className="ingestion-toggle">
            <input
              type="checkbox"
              checked={Boolean(toggle.value)}
              onChange={(event) => onToggle(toggle.path, event.target.checked)}
              disabled={busy}
            />
            <div>
              <strong>{toggle.label}</strong>
              <span>{toggle.description}</span>
            </div>
          </label>
        ))}
      </div>
      <div className="ingestion-actions">
        <p className="muted ingestion-dirty-hint">
          {hasPending
            ? "Changes pending — select Save to apply."
            : "Ingestion settings are up to date."}
        </p>
        <button
          type="button"
          className="primary small"
          onClick={onSave}
          disabled={busy || !hasPending}
        >
          Save ingestion settings
        </button>
      </div>
    </section>
  );
}

function SettingsPanel({ apiFetch, sessionActive, notify }) {
  const [settings, setSettings] = useState({});
  const [savedSettings, setSavedSettings] = useState({});
  const [dirtySettings, setDirtySettings] = useState({});
  const [meta, setMeta] = useState(createDefaultMeta());
  const [secrets, setSecrets] = useState({});
  const [schema, setSchema] = useState(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [activeCategory, setActiveCategory] = useState("__general__");

  const resetState = useCallback(() => {
    setSettings({});
    setSavedSettings({});
    setDirtySettings({});
    setMeta(createDefaultMeta());
    setSecrets({});
    setSchema(null);
    setSchemaError("");
    setError("");
  }, []);

  const loadSchema = useCallback(async () => {
    if (!sessionActive) {
      setSchema(null);
      setSchemaError("");
      return;
    }
    setSchemaLoading(true);
    setSchemaError("");
    try {
      const response = await apiFetch("/settings/schema");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setSchemaError(data.message || "Unable to load schema");
        return;
      }
      setSchema(data.schema || data);
    } catch (fetchError) {
      if (fetchError.code !== "NO_SESSION") {
        setSchemaError("Unable to load schema");
      }
    } finally {
      setSchemaLoading(false);
    }
  }, [apiFetch, sessionActive]);

  const loadSettings = useCallback(async () => {
    if (!sessionActive) {
      resetState();
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch("/settings");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(data.message || "Unable to load settings");
        return;
      }
      const loadedSettings = data.settings || {};
      setSettings(loadedSettings);
      setSavedSettings(loadedSettings);
      setDirtySettings({});
      setMeta(data.meta || createDefaultMeta());
      setSecrets(data.secrets || {});
    } catch (fetchError) {
      if (fetchError.code !== "NO_SESSION") {
        setError("Unable to load settings");
      }
    } finally {
      setLoading(false);
    }
  }, [apiFetch, sessionActive, resetState]);

  useEffect(() => {
    if (!sessionActive) {
      resetState();
      return;
    }
    loadSchema();
    loadSettings();
  }, [sessionActive, loadSchema, loadSettings, resetState]);

  const updateSetting = useCallback((path, value) => {
    setSettings((current) => updateNestedValue(current, path, value));
    const key = path.join(".");
    setDirtySettings((current) => ({ ...current, [key]: value }));
  }, []);

  const saveAllSettings = useCallback(async () => {
    if (!sessionActive || !Object.keys(dirtySettings).length) {
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await apiFetch("/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dirtySettings),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = data.message || "Unable to save settings";
        setError(message);
        if (notify) {
          notify({ type: "error", message });
        }
        return;
      }
      if (data.settings) {
        setSettings(data.settings);
        setSavedSettings(data.settings);
      }
      if (data.meta) {
        setMeta(data.meta);
      }
      if (data.secrets) {
        setSecrets(data.secrets);
      }
      setDirtySettings({});
      if (notify) {
        notify({ type: "success", message: "Settings saved successfully" });
      }
    } catch (saveError) {
      if (saveError.code !== "NO_SESSION") {
        const message = "Unable to save settings";
        setError(message);
        if (notify) {
          notify({ type: "error", message });
        }
      }
    } finally {
      setBusy(false);
    }
  }, [apiFetch, notify, sessionActive, dirtySettings]);

  const categories = useMemo(() => buildCategories(schema || {}), [schema]);

  useEffect(() => {
    if (!categories.length) {
      setActiveCategory("__general__");
      return;
    }
    if (!categories.some((category) => category.key === activeCategory)) {
      setActiveCategory(categories[0].key);
    }
  }, [categories, activeCategory]);

  const activeSchema = categories.find(
    (category) => category.key === activeCategory,
  );

  if (!sessionActive) {
    return (
      <section className="section">
        <h2>System Settings</h2>
        <div className="card placeholder-card">
          <p>Sign in to review and update ingestion, clustering, and integrations.</p>
        </div>
      </section>
    );
  }

  const databaseSummary =
    meta?.database?.summary ||
    "Database info unavailable (sign in to view details)";
  const syncMeta = meta?.sync || {};
  const migrationsMeta = meta?.migrations || {};
  const capabilityList = Array.isArray(meta?.capabilities)
    ? meta.capabilities
    : [];
  const capabilityIssues = capabilityList.filter(
    (item) => item && item.enabled && item.installed === false,
  );
  const capabilitySuccess =
    capabilityList.length > 0 && capabilityIssues.length === 0;

  return (
    <section className="section">
      <h2>System Settings</h2>
      <div className="card settings-card">
        <div className="card-header settings-header">
          <div className="settings-header-text">
            <h3>Runtime controls</h3>
            <p className="card-subtitle">
              Adjust clustering, ingestion, and integration behaviour in real time.
            </p>
            <p className="database-banner">{databaseSummary}</p>
            {syncMeta.enabled ? (
              <p className="muted">
                Sync backend {syncMeta.backend || "unknown"} targeting{' '}
                {syncMeta.connection || "n/a"}.
              </p>
            ) : (
              <p className="muted">Sync is currently disabled.</p>
            )}
            {migrationsMeta.error ? (
              <p className="error-text">
                Structural migrations failed: {migrationsMeta.error}
              </p>
            ) : migrationsMeta.skipped ? (
              <p className="muted">
                Automatic schema upgrades are disabled.{' '}
                {migrationsMeta.message ||
                  "Run the migration scripts manually when ready."}
              </p>
            ) : migrationsMeta.last_run ? (
              <p className="muted">
                Structural migrations last checked at {migrationsMeta.last_run}
                {migrationsMeta.message
                  ? ` (${migrationsMeta.message})`
                  : ''}.
              </p>
            ) : migrationsMeta.message ? (
              <p className="muted">{migrationsMeta.message}</p>
            ) : null}
            {capabilityIssues.length ? (
              <div className="capability-alert">
                <p className="error-text">
                  {capabilityIssues.length} provider capability issue
                  {capabilityIssues.length > 1 ? "s" : ""} detected:
                </p>
                <ul>
                  {capabilityIssues.map((issue) => (
                    <li key={issue.key || issue.label}>
                      <strong>{issue.label || issue.key}:</strong>{' '}
                      {issue.message}{' '}
                      {issue.resolution}
                    </li>
                  ))}
                </ul>
              </div>
            ) : capabilitySuccess ? (
              <p className="muted">
                All configured provider integrations are available.
              </p>
            ) : null}
          </div>
          <div className="button-row inline">
            <button
              type="button"
              className="secondary small"
              onClick={loadSchema}
              disabled={schemaLoading || busy}
            >
              {schemaLoading ? "Refreshing schema…" : "Reload schema"}
            </button>
            <button
              type="button"
              className="secondary small"
              onClick={loadSettings}
              disabled={loading || busy}
            >
              {loading ? "Refreshing…" : "Refresh"}
            </button>
            <button
              type="button"
              className="primary small"
              onClick={saveAllSettings}
              disabled={busy || Object.keys(dirtySettings).length === 0}
            >
              Save Changes
            </button>
          </div>
        </div>
        <IngestionModeSelector
          settings={settings}
          dirtySettings={dirtySettings}
          onToggle={updateSetting}
          onSave={saveAllSettings}
          busy={busy}
        />
        {schemaError ? <p className="error-text">{schemaError}</p> : null}
        {error ? <p className="error-text">{error}</p> : null}
        {schemaLoading && !schema ? (
          <p className="status-text">Loading schema…</p>
        ) : null}
        {loading && !Object.keys(settings || {}).length ? (
          <p className="status-text">Loading settings…</p>
        ) : null}
        <SettingsTabs
          categories={categories}
          activeKey={activeCategory}
          onSelect={setActiveCategory}
          disabled={busy || schemaLoading}
        />
        <div className="settings-body">
          {activeSchema ? (
            <SettingsCategoryPanel
              category={activeSchema}
              settings={settings}
              onUpdate={updateSetting}
              busy={busy}
              secretState={secrets}
            />
          ) : (
            <p className="status-text">Schema unavailable.</p>
          )}
        </div>
      </div>
    </section>
  );
}

function SettingsTabs({ categories, activeKey, onSelect, disabled }) {
  if (!Array.isArray(categories) || !categories.length) {
    return null;
  }
  return (
    <div className="settings-tabs" role="tablist">
      {categories.map((category) => {
        const isActive = category.key === activeKey;
        return (
          <button
            type="button"
            key={category.key}
            className={`settings-tab${isActive ? " active" : ""}`}
            onClick={() => onSelect(category.key)}
            disabled={disabled || isActive}
          >
            {category.title}
          </button>
        );
      })}
    </div>
  );
}

function SettingsCategoryPanel({
  category,
  settings,
  onUpdate,
  busy,
  secretState,
}) {
  if (!category) {
    return <p className="status-text">Schema unavailable.</p>;
  }
  const basePath = category.key === "__general__" ? [] : [category.key];
  const description = category.description;
  return (
    <div className="settings-category-panel">
      {description ? <p className="control-description">{description}</p> : null}
      <SettingsCategoryView
        schema={category.schema}
        basePath={basePath}
        settings={settings}
        onUpdate={onUpdate}
        busy={busy}
        secretState={secretState}
      />
    </div>
  );
}

function SettingsCategoryView({
  schema,
  basePath,
  settings,
  onUpdate,
  busy,
  secretState,
}) {
  if (!schema || schema.type !== "object" || !schema.properties) {
    return <p className="muted">No editable fields in this section.</p>;
  }
  return (
    <div className="settings-category">
      {Object.entries(schema.properties).map(([key, descriptor]) => {
        const path = [...basePath, key];
        const value = getNestedValue(settings, path);
        return (
          <SettingsNode
            key={path.join(".")}
            path={path}
            schema={descriptor}
            value={value}
            settings={settings}
            onUpdate={onUpdate}
            busy={busy}
            secretState={secretState}
          />
        );
      })}
    </div>
  );
}

function SettingsNode({
  path,
  schema,
  value,
  settings,
  onUpdate,
  busy,
  secretState,
}) {
  if (
    schema &&
    schema.type === "object" &&
    schema.properties &&
    Object.keys(schema.properties).length
  ) {
    return (
      <SettingsSubsection
        path={path}
        schema={schema}
        settings={settings}
        onUpdate={onUpdate}
        busy={busy}
        secretState={secretState}
      />
    );
  }

  return (
    <SettingsFieldControl
      path={path}
      schema={schema}
      value={value}
      onUpdate={onUpdate}
      busy={busy}
      secretState={secretState}
    />
  );
}

function SettingsSubsection({
  path,
  schema,
  settings,
  onUpdate,
  busy,
  secretState,
}) {
  const title = schema?.title || formatLabel(path[path.length - 1]);
  const description = schema?.description || "";
  return (
    <details className="settings-subsection" open>
      <summary>
        <span>{title}</span>
        {description ? <p className="muted">{description}</p> : null}
      </summary>
      <div className="settings-subsection-content">
        <SettingsCategoryView
          schema={schema}
          basePath={path}
          settings={settings}
          onUpdate={onUpdate}
          busy={busy}
          secretState={secretState}
        />
      </div>
    </details>
  );
}

function SettingsFieldControl({ path, schema, value, onUpdate, busy, secretState }) {
  const label = schema?.title || formatLabel(path[path.length - 1]);
  const description = schema?.description || "";
  const fieldType = schema?.type;
  const dotPath = path.join(".");
  const inputId = `setting-${path.join("-")}`;
  const isSecret = isSecretField(schema);
  const hasStoredSecret = Boolean(secretState && secretState[dotPath]);
  const enumOptions = extractEnumOptions(schema);
  const nullable = isNullableField(schema);
  const isBoolean = fieldType === "boolean";
  const isArrayField = fieldType === "array";
  const isObjectField = fieldType === "object" && !schema?.properties;
  const [localValue, setLocalValue] = useState(
    serialiseFieldValue(schema, value, { isSecret, hasStoredSecret }),
  );
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    if (isSecret && localValue) {
      return;
    }
    setLocalValue(serialiseFieldValue(schema, value, { isSecret, hasStoredSecret }));
    setErrorMessage("");
  }, [value, isSecret, hasStoredSecret, schema]);

  const handleChange = (event) => {
    setLocalValue(event.target.value);
    if (errorMessage) {
      setErrorMessage("");
    }
  };

  const handleBlur = () => {
    setErrorMessage("");
    try {
      const parsed = parseFieldDraft(schema, localValue, { isSecret });
      onUpdate(path, parsed);
    } catch (error) {
      setErrorMessage(error.message || "Invalid value");
    }
  };

  if (isBoolean) {
    return (
      <label className="setting-control">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onUpdate(path, event.target.checked)}
          disabled={busy}
        />
        <div>
          <span className="control-title">{label}</span>
          {description ? (
            <p className="control-description">{description}</p>
          ) : null}
        </div>
      </label>
    );
  }

  if (enumOptions) {
    const currentKey = value === null || value === undefined ? "" : String(value);
    return (
      <label className="setting-control" htmlFor={inputId}>
        <span className="control-title">{label}</span>
        {description ? <p className="control-description">{description}</p> : null}
        <select
          id={inputId}
          value={currentKey}
          onChange={(event) => {
            const selectedKey = event.target.value;
            if (!selectedKey && nullable) {
              onUpdate(path, null);
              return;
            }
            const match = enumOptions.find((option) => option.key === selectedKey);
            const nextValue = match ? match.value : selectedKey;
            onUpdate(path, nextValue);
          }}
          disabled={busy}
        >
          {nullable ? <option value="">Not set</option> : null}
          {enumOptions.map((option) => (
            <option key={option.key} value={option.key}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
    );
  }

  const isJsonField = isArrayField || isObjectField;

  return (
    <div className="setting-control vertical">
      <label className="control-title" htmlFor={inputId}>
        {label}
      </label>
      {description ? <p className="control-description">{description}</p> : null}
      {isSecret && hasStoredSecret ? (
        <p className="muted">A value is stored securely for this field. Enter a new value to update.</p>
      ) : null}
      {isJsonField ? (
        <textarea
          id={inputId}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={handleBlur}
          rows={schema?.type === "array" ? 4 : 5}
          disabled={busy}
        />
      ) : (
        <input
          id={inputId}
          type={isSecret ? "password" : schema?.type === "number" || schema?.type === "integer" ? "number" : "text"}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={handleBlur}
          autoComplete={isSecret ? "new-password" : "off"}
          disabled={busy}
        />
      )}
      {errorMessage ? <p className="error-text">{errorMessage}</p> : null}
    </div>
  );
}

function serialiseFieldValue(schema, value, { isSecret, hasStoredSecret }) {
  if (isSecret) {
    return "";
  }
  if (value === undefined || value === null) {
    return "";
  }
  if (schema?.type === "array" || (schema?.type === "object" && !schema?.properties)) {
    try {
      return JSON.stringify(value, null, 2);
    } catch (error) {
      return "";
    }
  }
  return String(value);
}

function parseFieldDraft(schema, draft, { isSecret }) {
  if (isSecret) {
    if (typeof draft === "string") {
      return draft.trim();
    }
    return draft;
  }

  const type = schema?.type;
  const text = typeof draft === "string" ? draft : "";
  const nullable = isNullableField(schema);

  if (type === "integer") {
    if (!text.trim()) {
      if (nullable) {
        return null;
      }
      throw new Error("Enter an integer value");
    }
    const parsed = Number.parseInt(text, 10);
    if (Number.isNaN(parsed)) {
      throw new Error("Enter a valid integer");
    }
    return parsed;
  }

  if (type === "number") {
    if (!text.trim()) {
      if (nullable) {
        return null;
      }
      throw new Error("Enter a numeric value");
    }
    const parsed = Number.parseFloat(text);
    if (Number.isNaN(parsed)) {
      throw new Error("Enter a numeric value");
    }
    return parsed;
  }

  if (type === "array" || (type === "object" && !schema?.properties)) {
    if (!text.trim()) {
      if (nullable) {
        return null;
      }
      return type === "array" ? [] : {};
    }
    try {
      const parsed = JSON.parse(text);
      if (type === "array" && !Array.isArray(parsed)) {
        throw new Error("Enter a JSON array");
      }
      if (
        type === "object" &&
        (parsed === null || typeof parsed !== "object" || Array.isArray(parsed))
      ) {
        throw new Error("Enter a JSON object");
      }
      return parsed;
    } catch (error) {
      throw new Error("Enter valid JSON");
    }
  }

  if (!text.trim() && nullable) {
    return null;
  }

  return text;
}

function Rebuild({ apiFetch, sessionActive }) {
  const [mode, setMode] = useState("vector");
  const [status, setStatus] = useState("");

  const rebuild = async () => {
    setStatus("Rebuilding...");
    try {
      const response = await apiFetch(`/clusters?mode=${mode}`, {
        method: "POST",
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setStatus(data.message || "Failed to rebuild");
        return;
      }
      setStatus("Cluster index rebuilt");
    } catch (error) {
      if (error.code === "NO_SESSION") {
        setStatus("Login required");
        return;
      }
      setStatus("Error rebuilding clusters");
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3>Rebuild Clusters</h3>
        <p className="card-subtitle">
          Refresh heuristics or vector embeddings.
        </p>
      </div>
      <div className="card-controls">
        <select value={mode} onChange={(event) => setMode(event.target.value)}>
          <option value="vector">Vector</option>
          <option value="heuristic">Heuristic</option>
        </select>
        <button
          onClick={rebuild}
          className="primary"
          disabled={!sessionActive || status === "Rebuilding..."}
        >
          Rebuild
        </button>
      </div>
      {status ? <span className="status-text">{status}</span> : null}
    </div>
  );
}

function Activity({ apiFetch, sessionActive }) {
  const [params, setParams] = useState({ top_n: "5", fading_threshold: "" });
  const [activity, setActivity] = useState({ active: [], fading: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const update = (key, value) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const fetchActivity = useCallback(async () => {
    if (!sessionActive) {
      return;
    }
    setLoading(true);
    setError("");
    const query = Object.entries(params)
      .filter(
        ([_, value]) => value !== undefined && value !== null && value !== "",
      )
      .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
      .join("&");
    try {
      const response = await apiFetch(
        `/clusters/activity${query ? `?${query}` : ""}`,
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(data.message || "Unable to load activity");
        setActivity({ active: [], fading: [] });
        return;
      }
      setActivity({
        active: Array.isArray(data.active) ? data.active : [],
        fading: Array.isArray(data.fading) ? data.fading : [],
      });
    } catch (error) {
      if (error.code !== "NO_SESSION") {
        setError("Unable to load activity");
      }
    } finally {
      setLoading(false);
    }
  }, [apiFetch, params, sessionActive]);

  useEffect(() => {
    fetchActivity();
  }, [fetchActivity]);

  return (
    <div className="card">
      <div className="card-header">
        <h3>Cluster Activity</h3>
        <p className="card-subtitle">
          Quick summary of recent and fading clusters.
        </p>
      </div>
      <div className="card-controls inline">
        <label>
          Top N
          <input
            type="number"
            min="1"
            value={params.top_n}
            onChange={(event) => update("top_n", event.target.value)}
          />
        </label>
        <label>
          Fading threshold
          <input
            type="number"
            value={params.fading_threshold}
            onChange={(event) => update("fading_threshold", event.target.value)}
          />
        </label>
        <button
          className="secondary"
          onClick={fetchActivity}
          disabled={!sessionActive || loading}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="activity-columns">
        <div>
          <h4>Active</h4>
          {activity.active.length ? (
            <ul>
              {activity.active.map((item, index) => (
                <li key={index}>{item.summary || JSON.stringify(item)}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No active clusters.</p>
          )}
        </div>
        <div>
          <h4>Fading</h4>
          {activity.fading.length ? (
            <ul>
              {activity.fading.map((item, index) => (
                <li key={index}>{item.summary || JSON.stringify(item)}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No fading clusters.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function ClusterList({ apiFetch, sessionActive }) {
  const [params, setParams] = useState({
    keyword: "",
    min_size: "",
    max_size: "",
    min_weight: "",
    max_weight: "",
    min_polarity: "",
    max_polarity: "",
  });
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const chartCanvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  const update = (key, value) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const fetchClusters = useCallback(async () => {
    if (!sessionActive) {
      return;
    }
    setLoading(true);
    setError("");
    const query = Object.entries(params)
      .filter(
        ([_, value]) => value !== undefined && value !== null && value !== "",
      )
      .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
      .join("&");
    try {
      const response = await apiFetch(`/clusters${query ? `?${query}` : ""}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(data.message || "Unable to load clusters");
        setClusters([]);
        return;
      }
      const list = Array.isArray(data.clusters) ? data.clusters : [];
      setClusters(list);
    } catch (error) {
      if (error.code !== "NO_SESSION") {
        setError("Unable to load clusters");
      }
    } finally {
      setLoading(false);
    }
  }, [apiFetch, params, sessionActive]);

  useEffect(() => {
    fetchClusters();
  }, [fetchClusters]);

  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }
    if (!chartCanvasRef.current || !clusters.length) {
      return;
    }
    const ctx = chartCanvasRef.current.getContext("2d");
    const labels = clusters.map(
      (cluster, index) =>
        cluster.id ||
        cluster.cluster_id ||
        cluster.label ||
        `cluster-${index + 1}`,
    );
    const dataset = (key) =>
      clusters.map((cluster) => {
        const numeric = Number(cluster[key]);
        return Number.isFinite(numeric) ? numeric : 0;
      });
    chartInstanceRef.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Size",
            data: dataset("size"),
            backgroundColor: "rgba(54, 162, 235, 0.5)",
          },
          {
            label: "Weight",
            data: dataset("weight"),
            backgroundColor: "rgba(255, 159, 64, 0.5)",
          },
          {
            label: "Importance",
            data: dataset("importance"),
            backgroundColor: "rgba(75, 192, 192, 0.5)",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
        },
        scales: {
          x: {
            ticks: { color: "#374151" },
            grid: { color: "rgba(107, 114, 128, 0.15)" },
          },
          y: {
            ticks: { color: "#374151" },
            grid: { color: "rgba(107, 114, 128, 0.15)" },
          },
        },
      },
    });
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [clusters]);

  return (
    <div className="card">
      <div className="card-header">
        <h3>Cluster Catalog</h3>
        <p className="card-subtitle">
          Inspect cluster statistics across filters.
        </p>
      </div>
      <div className="cluster-filters">
        <input
          placeholder="Keyword"
          value={params.keyword}
          onChange={(event) => update("keyword", event.target.value)}
        />
        <input
          placeholder="Min size"
          type="number"
          value={params.min_size}
          onChange={(event) => update("min_size", event.target.value)}
        />
        <input
          placeholder="Max size"
          type="number"
          value={params.max_size}
          onChange={(event) => update("max_size", event.target.value)}
        />
        <input
          placeholder="Min weight"
          type="number"
          value={params.min_weight}
          onChange={(event) => update("min_weight", event.target.value)}
        />
        <input
          placeholder="Max weight"
          type="number"
          value={params.max_weight}
          onChange={(event) => update("max_weight", event.target.value)}
        />
        <input
          placeholder="Min polarity"
          type="number"
          value={params.min_polarity}
          onChange={(event) => update("min_polarity", event.target.value)}
        />
        <input
          placeholder="Max polarity"
          type="number"
          value={params.max_polarity}
          onChange={(event) => update("max_polarity", event.target.value)}
        />
        <button
          className="secondary"
          onClick={fetchClusters}
          disabled={!sessionActive || loading}
        >
          {loading ? "Loading..." : "Apply"}
        </button>
      </div>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Size</th>
              <th>Weight</th>
              <th>Importance</th>
            </tr>
          </thead>
          <tbody>
            {clusters.length ? (
              clusters.map((cluster) => (
                <tr key={cluster.id || cluster.cluster_id}>
                  <td>{cluster.id || cluster.cluster_id}</td>
                  <td>{cluster.size}</td>
                  <td>{cluster.weight}</td>
                  <td>{cluster.importance}</td>
                </tr>
              ))
            ) : (
              <tr className="empty-row">
                <td colSpan={4}>
                  {loading ? "Loading clusters..." : "No clusters found"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="cluster-chart">
        <canvas ref={chartCanvasRef} height="200" />
      </div>
    </div>
  );
}
function renderCellValue(value) {
  if (value === null || value === undefined) {
    return <span className="muted">—</span>;
  }
  if (typeof value === "object") {
    try {
      const text = JSON.stringify(value);
      const display =
        text.length > MAX_CELL_LENGTH
          ? `${text.slice(0, MAX_CELL_LENGTH - 1)}…`
          : text;
      return <span title={text}>{display}</span>;
    } catch (error) {
      return <span>{String(value)}</span>;
    }
  }
  const text = String(value);
  if (text.length > MAX_CELL_LENGTH) {
    return <span title={text}>{text.slice(0, MAX_CELL_LENGTH - 1)}…</span>;
  }
  return text;
}

function stringifyValue(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (error) {
      return String(value);
    }
  }
  return String(value);
}

function shouldUseTextarea(column) {
  const type = (column.type || "").toLowerCase();
  return (
    type.includes("text") ||
    type.includes("json") ||
    type.includes("char") ||
    type.includes("clob") ||
    type.includes("blob")
  );
}

function buildInitialFormValues(columns, row, mode) {
  const values = {};
  columns.forEach((column) => {
    if (mode === "add") {
      values[column.name] = "";
      return;
    }
    if (row && Object.prototype.hasOwnProperty.call(row, column.name)) {
      const value = row[column.name];
      if (value === null || value === undefined) {
        values[column.name] = "";
      } else if (typeof value === "object") {
        try {
          values[column.name] = JSON.stringify(value);
        } catch (error) {
          values[column.name] = String(value);
        }
      } else {
        values[column.name] = String(value);
      }
    } else {
      values[column.name] = "";
    }
  });
  return values;
}

function validateFormValues(formValues, columns, mode) {
  const errors = {};
  columns.forEach((column) => {
    if (mode === "edit" && column.primary_key) {
      return;
    }
    const hasDefault =
      column.default !== null && column.default !== undefined;
    const required =
      !column.nullable &&
      !hasDefault &&
      !(mode === "add" && column.primary_key);
    if (!required) {
      return;
    }
    const value = formValues[column.name];
    if (
      value === undefined ||
      value === null ||
      (typeof value === "string" && value.trim() === "")
    ) {
      errors[column.name] = "Required";
    }
  });
  return errors;
}

function coerceFormValues(formValues, columns, mode) {
  const values = {};
  const errors = {};
  columns.forEach((column) => {
    const raw = formValues[column.name];
    const isString = typeof raw === "string";
    const trimmed = isString ? raw.trim() : raw;
    const isBlank =
      raw === undefined ||
      raw === null ||
      (isString && trimmed === "");
    if (isBlank) {
      if (mode === "edit") {
        values[column.name] = null;
      }
      return;
    }

    const type = (column.type || "").toLowerCase();
    if (type.includes("json")) {
      if (!isString) {
        values[column.name] = raw;
        return;
      }
      try {
        values[column.name] = JSON.parse(raw);
      } catch (error) {
        errors[column.name] = "Invalid JSON";
      }
      return;
    }

    if (type.includes("bool")) {
      if (!isString) {
        values[column.name] = Boolean(raw);
        return;
      }
      const normalized = trimmed.toLowerCase();
      if (["true", "1", "yes", "on"].includes(normalized)) {
        values[column.name] = true;
      } else if (["false", "0", "no", "off"].includes(normalized)) {
        values[column.name] = false;
      } else {
        errors[column.name] = "Enter true or false";
      }
      return;
    }

    if (type.includes("int")) {
      const numberValue = Number(trimmed);
      if (!Number.isFinite(numberValue) || !Number.isInteger(numberValue)) {
        errors[column.name] = "Enter an integer";
        return;
      }
      values[column.name] = numberValue;
      return;
    }

    if (
      type.includes("real") ||
      type.includes("float") ||
      type.includes("double") ||
      type.includes("decimal") ||
      type.includes("numeric")
    ) {
      const numberValue = Number(trimmed);
      if (!Number.isFinite(numberValue)) {
        errors[column.name] = "Enter a number";
        return;
      }
      values[column.name] = numberValue;
      return;
    }

    values[column.name] = isString ? raw : String(raw);
  });

  return { values, errors };
}

function RowEditorModal({
  mode,
  columns,
  row,
  busy,
  error,
  onCancel,
  onSubmit,
}) {
  const [formValues, setFormValues] = useState(() =>
    buildInitialFormValues(columns, row, mode),
  );
  const [fieldErrors, setFieldErrors] = useState({});

  useEffect(() => {
    setFormValues(buildInitialFormValues(columns, row, mode));
    setFieldErrors({});
  }, [columns, row, mode]);

  const handleChange = useCallback((columnName, value) => {
    setFormValues((prev) => ({ ...prev, [columnName]: value }));
  }, []);

  const handleSubmit = useCallback(
    (event) => {
      event.preventDefault();
      const validationErrors = validateFormValues(formValues, columns, mode);
      const { values, errors } = coerceFormValues(formValues, columns, mode);
      const mergedErrors = { ...validationErrors, ...errors };
      if (Object.keys(mergedErrors).length) {
        setFieldErrors(mergedErrors);
        return;
      }
      setFieldErrors({});
      const submission = { ...values };
      if (mode === "edit") {
        columns
          .filter((column) => column.primary_key)
          .forEach((column) => {
            delete submission[column.name];
          });
      }
      onSubmit(submission);
    },
    [columns, formValues, mode, onSubmit],
  );

  return (
    <div className="modal-backdrop">
      <div className="modal modal-wide">
        <h2>{mode === "add" ? "Add row" : "Edit row"}</h2>
        <form className="row-form" onSubmit={handleSubmit}>
          <div className="row-form-grid">
            {columns.map((column, index) => {
              const hintParts = [];
              if (column.type) {
                hintParts.push(column.type);
              }
              if (mode !== "add" && column.primary_key) {
                hintParts.push("primary key");
              }
              if (!column.nullable && !(mode === "add" && column.primary_key)) {
                hintParts.push("required");
              }
              if (column.default) {
                hintParts.push(`default ${column.default}`);
              }
              const hintText = hintParts.join(" • ");
              const inputId = `${column.name}-${mode}`;
              const value = formValues[column.name] ?? "";
              const fieldError = fieldErrors[column.name];
              const disabled = busy || (mode === "edit" && column.primary_key);
              const shouldAutoFocus =
                index === 0 && !(mode === "edit" && column.primary_key);
              return (
                <label className="modal-label field-label" key={column.name}>
                  <span className="field-title">
                    {column.name}
                    {!column.nullable &&
                    !(mode === "add" && column.primary_key) ? (
                      <span className="required-indicator" aria-hidden="true">
                        *
                      </span>
                    ) : null}
                  </span>
                  {hintText ? (
                    <span className="field-hint">{hintText}</span>
                  ) : null}
                  {shouldUseTextarea(column) ? (
                    <textarea
                      id={inputId}
                      value={value}
                      onChange={(event) =>
                        handleChange(column.name, event.target.value)
                      }
                      disabled={disabled}
                      rows={3}
                      autoFocus={shouldAutoFocus}
                    />
                  ) : (
                    <input
                      id={inputId}
                      type="text"
                      value={value}
                      onChange={(event) =>
                        handleChange(column.name, event.target.value)
                      }
                      disabled={disabled}
                      autoFocus={shouldAutoFocus}
                    />
                  )}
                  {fieldError ? (
                    <span className="input-error">{fieldError}</span>
                  ) : null}
                </label>
              );
            })}
          </div>
          {error ? <p className="modal-error">{error}</p> : null}
          <div className="modal-actions">
            <button
              type="button"
              className="secondary"
              onClick={onCancel}
              disabled={busy}
            >
              Cancel
            </button>
            <button type="submit" className="primary" disabled={busy}>
              {busy
                ? mode === "add"
                  ? "Inserting..."
                  : "Saving..."
                : mode === "add"
                ? "Insert row"
                : "Save changes"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function DeleteRowModal({ columns, row, busy, error, onCancel, onConfirm }) {
  const primaryKeys = useMemo(
    () => columns.filter((column) => column.primary_key),
    [columns],
  );

  const preview = useMemo(() => {
    if (!row) {
      return [];
    }
    const limitedColumns = columns.slice(0, 6);
    return limitedColumns.map((column) => ({
      name: column.name,
      value: stringifyValue(row[column.name]),
    }));
  }, [columns, row]);

  return (
    <div className="modal-backdrop">
      <div className="modal modal-wide">
        <h2>Delete row</h2>
        <p className="modal-description">
          This action cannot be undone. The row will be permanently removed
          from the table.
        </p>
        {primaryKeys.length ? (
          <div className="row-preview">
            <h4>Primary key</h4>
            <ul>
              {primaryKeys.map((column) => (
                <li key={column.name}>
                  <strong>{column.name}:</strong> {stringifyValue(row[column.name])}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        <div className="row-preview">
          <h4>Row snapshot</h4>
          <ul>
            {preview.map((item) => (
              <li key={item.name}>
                <strong>{item.name}:</strong> {item.value}
              </li>
            ))}
          </ul>
        </div>
        {error ? <p className="modal-error">{error}</p> : null}
        <div className="modal-actions">
          <button
            type="button"
            className="secondary"
            onClick={onCancel}
            disabled={busy}
          >
            Cancel
          </button>
          <button
            type="button"
            className="danger"
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}

function TableBrowser({ apiFetch, sessionActive }) {
  const [tables, setTables] = useState([]);
  const [loadingTables, setLoadingTables] = useState(false);
  const [tablesError, setTablesError] = useState("");
  const [selectedTable, setSelectedTable] = useState(null);
  const [rowRequest, setRowRequest] = useState({
    page: 1,
    pageSize: DEFAULT_PAGE_SIZE,
    search: "",
    namespace: "all",
  });
  const [rowState, setRowState] = useState(() => buildEmptyRowState());
  const [rowLoading, setRowLoading] = useState(false);
  const [rowError, setRowError] = useState("");
  const [actionState, setActionState] = useState({ mode: null, row: null });
  const [actionError, setActionError] = useState("");
  const [actionBusy, setActionBusy] = useState(false);
  const fetchTokenRef = useRef(0);

  const refreshTables = useCallback(async () => {
    if (!sessionActive) {
      return;
    }
    setLoadingTables(true);
    setTablesError("");
    try {
      const response = await apiFetch("/admin/tables");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setTables([]);
        setTablesError(data.message || "Unable to load tables");
        return;
      }
      const list = Array.isArray(data.tables) ? data.tables : [];
      setTables(list);
      setSelectedTable((current) => {
        if (current && list.some((table) => table.name === current)) {
          return current;
        }
        return list.length ? list[0].name : null;
      });
    } catch (error) {
      if (error.code !== "NO_SESSION") {
        setTablesError("Unable to load tables");
      }
    } finally {
      setLoadingTables(false);
    }
  }, [apiFetch, sessionActive]);

  useEffect(() => {
    if (sessionActive) {
      refreshTables();
    } else {
      fetchTokenRef.current += 1;
      setTables([]);
      setSelectedTable(null);
      setRowRequest({
        page: 1,
        pageSize: DEFAULT_PAGE_SIZE,
        search: "",
        namespace: "all",
      });
      setRowState(buildEmptyRowState());
      setRowError("");
      setRowLoading(false);
      setActionState({ mode: null, row: null });
      setActionError("");
      setActionBusy(false);
    }
  }, [sessionActive, refreshTables]);

  useEffect(() => {
    if (!selectedTable) {
      return;
    }
    fetchTokenRef.current += 1;
    setActionState({ mode: null, row: null });
    setActionError("");
    setActionBusy(false);
    setRowRequest((prev) => ({
      page: 1,
      pageSize: prev.pageSize,
      search: "",
      namespace: "all",
    }));
    setRowState((prev) => buildEmptyRowState(prev.pageSize));
  }, [selectedTable]);

  const fetchRows = useCallback(async () => {
    if (!sessionActive || !selectedTable) {
      return;
    }
    const token = ++fetchTokenRef.current;
    const tableName = selectedTable;
    const requestState = rowRequest;
    setRowLoading(true);
    setRowError("");
    const params = new URLSearchParams();
    params.set("page", String(requestState.page));
    params.set("page_size", String(requestState.pageSize));
    if (requestState.search) {
      params.set("search", requestState.search);
    }
    if (requestState.namespace && requestState.namespace !== "all") {
      params.append("namespace", requestState.namespace);
    }

    try {
      const response = await apiFetch(
        `/admin/tables/${encodeURIComponent(tableName)}/rows?${params.toString()}`,
      );
      const data = await response.json().catch(() => ({}));
      if (fetchTokenRef.current !== token || selectedTable !== tableName) {
        return;
      }
      if (!response.ok) {
        setRowState((prev) => ({
          ...buildEmptyRowState(prev.pageSize || requestState.pageSize),
          columns: prev.columns,
        }));
        setRowError(data.message || "Unable to load rows");
        return;
      }
      setRowState({
        columns: Array.isArray(data.columns) ? data.columns : [],
        rows: Array.isArray(data.rows) ? data.rows : [],
        totalRows: typeof data.total_rows === "number" ? data.total_rows : 0,
        hasNext: Boolean(data.has_next),
        hasPrevious: Boolean(data.has_previous),
        namespaces: Array.isArray(data.namespaces) ? data.namespaces : [],
        page: typeof data.page === "number" ? data.page : requestState.page,
        pageSize:
          typeof data.page_size === "number"
            ? data.page_size
            : requestState.pageSize,
      });
      setRowError("");
    } catch (error) {
      if (fetchTokenRef.current !== token || selectedTable !== tableName) {
        return;
      }
      if (error.code === "NO_SESSION") {
        return;
      }
      setRowState((prev) => ({
        ...buildEmptyRowState(prev.pageSize || requestState.pageSize),
        columns: prev.columns,
      }));
      setRowError("Unable to load rows");
    } finally {
      if (fetchTokenRef.current === token && selectedTable === tableName) {
        setRowLoading(false);
      }
    }
  }, [apiFetch, sessionActive, selectedTable, rowRequest]);

  useEffect(() => {
    fetchRows();
  }, [fetchRows]);

  const selectedTableMeta = useMemo(
    () => tables.find((table) => table.name === selectedTable) || null,
    [tables, selectedTable],
  );

  const primaryKeyColumns = useMemo(
    () =>
      rowState.columns
        .filter((column) => column.primary_key)
        .map((column) => column.name),
    [rowState.columns],
  );

  const canMutate = Boolean(
    selectedTableMeta && selectedTableMeta.type === "table",
  );
  const hasPrimaryKey = primaryKeyColumns.length > 0;
  const mutationDisabled = actionBusy || rowLoading;

  const namespaceOptions = useMemo(() => {
    if (rowState.namespaces.length) {
      return rowState.namespaces;
    }
    return selectedTableMeta && Array.isArray(selectedTableMeta.namespaces)
      ? selectedTableMeta.namespaces
      : [];
  }, [rowState.namespaces, selectedTableMeta]);

  const rangeLabel = useMemo(() => {
    const page = rowState.page || 1;
    const pageSize = rowState.pageSize || rowRequest.pageSize;
    if (!rowState.rows.length) {
      if (rowState.totalRows) {
        return `0 of ${rowState.totalRows}`;
      }
      return "No rows to display";
    }
    const start = (page - 1) * pageSize + 1;
    const end = start + rowState.rows.length - 1;
    return `Showing ${start}–${end} of ${rowState.totalRows}`;
  }, [rowState, rowRequest.pageSize]);

  const handleSelectTable = useCallback((name) => {
    setSelectedTable((current) => (current === name ? current : name));
  }, []);

  const buildPrimaryKeyMapping = useCallback(
    (row) => {
      if (!row || !primaryKeyColumns.length) {
        return null;
      }
      const mapping = {};
      for (const column of primaryKeyColumns) {
        if (!Object.prototype.hasOwnProperty.call(row, column)) {
          return null;
        }
        mapping[column] = row[column];
      }
      return mapping;
    },
    [primaryKeyColumns],
  );

  const closeActionModal = useCallback(() => {
    if (actionBusy) {
      return;
    }
    setActionState({ mode: null, row: null });
    setActionError("");
  }, [actionBusy]);

  const openAddRow = useCallback(() => {
    if (!canMutate) {
      return;
    }
    setActionBusy(false);
    setActionError("");
    setActionState({ mode: "add", row: null });
  }, [canMutate]);

  const openEditRow = useCallback(
    (row) => {
      if (!canMutate || !hasPrimaryKey) {
        return;
      }
      setActionBusy(false);
      setActionError("");
      setActionState({ mode: "edit", row });
    },
    [canMutate, hasPrimaryKey],
  );

  const openDeleteRow = useCallback(
    (row) => {
      if (!canMutate || !hasPrimaryKey) {
        return;
      }
      setActionBusy(false);
      setActionError("");
      setActionState({ mode: "delete", row });
    },
    [canMutate, hasPrimaryKey],
  );

  const submitAddRow = useCallback(
    async (values) => {
      if (!selectedTable || !canMutate) {
        return;
      }
      setActionBusy(true);
      setActionError("");
      try {
        const response = await apiFetch(
          `/admin/tables/${encodeURIComponent(selectedTable)}/rows`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ values }),
          },
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          setActionError(data.message || "Unable to insert row");
          return;
        }
        setActionState({ mode: null, row: null });
        await fetchRows();
      } catch (error) {
        if (error.code === "NO_SESSION") {
          setActionError("Session expired. Please sign in again.");
        } else {
          setActionError("Unable to insert row");
        }
      } finally {
        setActionBusy(false);
      }
    },
    [apiFetch, selectedTable, canMutate, fetchRows],
  );

  const submitEditRow = useCallback(
    async (values) => {
      if (!selectedTable || !canMutate || !actionState.row) {
        return;
      }
      const mapping = buildPrimaryKeyMapping(actionState.row);
      if (!mapping) {
        setActionError("Unable to resolve primary key for this row");
        return;
      }
      if (!Object.keys(values).length) {
        setActionError("Provide at least one value to update");
        return;
      }
      const pkSegment = encodeURIComponent(JSON.stringify(mapping));
      setActionBusy(true);
      setActionError("");
      try {
        const response = await apiFetch(
          `/admin/tables/${encodeURIComponent(selectedTable)}/rows/${pkSegment}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ values, primary_key: mapping }),
          },
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          setActionError(data.message || "Unable to update row");
          return;
        }
        setActionState({ mode: null, row: null });
        await fetchRows();
      } catch (error) {
        if (error.code === "NO_SESSION") {
          setActionError("Session expired. Please sign in again.");
        } else {
          setActionError("Unable to update row");
        }
      } finally {
        setActionBusy(false);
      }
    },
    [
      apiFetch,
      selectedTable,
      canMutate,
      actionState.row,
      buildPrimaryKeyMapping,
      fetchRows,
    ],
  );

  const confirmDeleteRow = useCallback(async () => {
    if (!selectedTable || !canMutate || !actionState.row) {
      return;
    }
    const mapping = buildPrimaryKeyMapping(actionState.row);
    if (!mapping) {
      setActionError("Unable to resolve primary key for this row");
      return;
    }
    const pkSegment = encodeURIComponent(JSON.stringify(mapping));
    setActionBusy(true);
    setActionError("");
    try {
      const response = await apiFetch(
        `/admin/tables/${encodeURIComponent(selectedTable)}/rows/${pkSegment}`,
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ primary_key: mapping }),
        },
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setActionError(data.message || "Unable to delete row");
        return;
      }
      const currentPage = rowState.page || rowRequest.page || 1;
      const shouldPageBack = rowState.rows.length <= 1 && currentPage > 1;
      setActionState({ mode: null, row: null });
      if (shouldPageBack) {
        setRowRequest((prev) => ({
          ...prev,
          page: Math.max(1, currentPage - 1),
        }));
      } else {
        await fetchRows();
      }
    } catch (error) {
      if (error.code === "NO_SESSION") {
        setActionError("Session expired. Please sign in again.");
      } else {
        setActionError("Unable to delete row");
      }
    } finally {
      setActionBusy(false);
    }
  }, [
    apiFetch,
    selectedTable,
    canMutate,
    actionState.row,
    buildPrimaryKeyMapping,
    rowState.rows.length,
    rowState.page,
    rowRequest.page,
    fetchRows,
  ]);

  if (!sessionActive) {
    return (
      <section className="section">
        <h2>Database Explorer</h2>
        <div className="card placeholder-card">
          <p>Sign in with your API key to browse database tables.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="section">
      <h2>Database Explorer</h2>
      <div className="table-browser">
        <aside className="table-list">
          <div className="table-list-header">
            <div>
              <h3>Tables</h3>
              <p className="muted">
                Select a table to inspect column metadata and rows.
              </p>
            </div>
            <button
              className="secondary small"
              onClick={refreshTables}
              disabled={loadingTables}
            >
              Refresh
            </button>
          </div>
          {tablesError ? <p className="error-text">{tablesError}</p> : null}
          {loadingTables ? <p className="muted">Loading tables...</p> : null}
          {!loadingTables && !tables.length ? (
            <p className="muted">No tables available.</p>
          ) : (
            <ul>
              {tables.map((table) => (
                <li
                  key={table.name}
                  className={table.name === selectedTable ? "selected" : ""}
                >
                  <button
                    type="button"
                    onClick={() => handleSelectTable(table.name)}
                  >
                    <span className="table-name">{table.name}</span>
                    <span className="table-meta">
                      {table.row_count !== null && table.row_count !== undefined
                        ? `${table.row_count} rows`
                        : "Unknown row count"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>
        <div className="table-content">
          {selectedTable ? (
            <div className="card table-panel">
              <div className="table-panel-header">
                <div>
                  <h3>{selectedTable}</h3>
                  <div className="table-panel-meta">
                    {selectedTableMeta && selectedTableMeta.type ? (
                      <span className="badge neutral">
                        {selectedTableMeta.type}
                      </span>
                    ) : null}
                    <span className="muted">
                      {rowState.totalRows
                        ? `${rowState.totalRows} total rows`
                        : "No rows"}
                    </span>
                  </div>
                </div>
                <div className="namespace-tags">
                  {namespaceOptions.slice(0, 6).map((namespace) => (
                    <span className="badge" key={namespace}>
                      {namespace}
                    </span>
                  ))}
                  {namespaceOptions.length > 6 ? (
                    <span className="badge neutral">
                      +{namespaceOptions.length - 6}
                    </span>
                  ) : null}
                </div>
              </div>
              <div className="table-filters">
                <input
                  type="search"
                  value={rowRequest.search}
                  placeholder="Search across columns"
                  onChange={(event) =>
                    setRowRequest((prev) => ({
                      ...prev,
                      page: 1,
                      search: event.target.value,
                    }))
                  }
                />
                <select
                  value={rowRequest.namespace}
                  onChange={(event) =>
                    setRowRequest((prev) => ({
                      ...prev,
                      page: 1,
                      namespace: event.target.value,
                    }))
                  }
                >
                  <option value="all">All namespaces</option>
                  {namespaceOptions.map((namespace) => (
                    <option key={namespace} value={namespace}>
                      {namespace}
                    </option>
                  ))}
                </select>
                <select
                  value={rowRequest.pageSize}
                  onChange={(event) => {
                    const size =
                      Number(event.target.value) || DEFAULT_PAGE_SIZE;
                    setRowRequest((prev) => ({
                      ...prev,
                      page: 1,
                      pageSize: size,
                    }));
                  }}
                >
                  {[10, 25, 50, 100, 200].map((size) => (
                    <option key={size} value={size}>
                      {size} / page
                    </option>
                  ))}
                </select>
                <button
                  className="secondary"
                  onClick={() =>
                    setRowRequest((prev) => ({
                      ...prev,
                      page: 1,
                      search: "",
                      namespace: "all",
                    }))
                  }
                  disabled={
                    !rowRequest.search && rowRequest.namespace === "all"
                  }
                >
                  Clear filters
                </button>
                {canMutate ? (
                  <button
                    type="button"
                    className="primary"
                    onClick={openAddRow}
                    disabled={mutationDisabled}
                  >
                    {actionBusy && actionState.mode === "add"
                      ? "Adding..."
                      : "Add row"}
                  </button>
                ) : null}
              </div>
              {rowError ? <p className="error-text">{rowError}</p> : null}
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      {rowState.columns.map((column) => (
                        <th key={column.name}>
                          <div className="column-name">{column.name}</div>
                          <div className="column-meta">
                            <span>{column.type || "unknown"}</span>
                            {!column.nullable ? <span>• not null</span> : null}
                            {column.primary_key ? <span>• primary</span> : null}
                          </div>
                        </th>
                      ))}
                      {canMutate ? <th className="actions-column">Actions</th> : null}
                    </tr>
                  </thead>
                  <tbody>
                    {rowState.rows.length ? (
                      rowState.rows.map((row, index) => {
                        const key =
                          row.id ||
                          row.memory_id ||
                          row.memoryId ||
                          row.uuid ||
                          `row-${index}`;
                        return (
                          <tr key={key}>
                            {rowState.columns.map((column) => (
                              <td key={column.name}>
                                {renderCellValue(row[column.name])}
                              </td>
                            ))}
                            {canMutate ? (
                              <td className="actions-cell">
                                {hasPrimaryKey ? (
                                  <div className="row-actions">
                                    <button
                                      type="button"
                                      className="secondary small"
                                      onClick={() => openEditRow(row)}
                                      disabled={mutationDisabled}
                                    >
                                      Edit
                                    </button>
                                    <button
                                      type="button"
                                      className="danger small"
                                      onClick={() => openDeleteRow(row)}
                                      disabled={mutationDisabled}
                                    >
                                      Delete
                                    </button>
                                  </div>
                                ) : (
                                  <span className="muted">No primary key</span>
                                )}
                              </td>
                            ) : null}
                          </tr>
                        );
                      })
                    ) : (
                      <tr className="empty-row">
                        <td
                          colSpan={
                            Math.max(
                              rowState.columns.length + (canMutate ? 1 : 0),
                              1,
                            )
                          }
                        >
                          {rowLoading
                            ? "Loading rows..."
                            : "No rows to display"}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="table-footer">
                <span className="muted">{rangeLabel}</span>
                <div className="pagination-controls">
                  <button
                    className="secondary"
                    onClick={() =>
                      setRowRequest((prev) => ({
                        ...prev,
                        page: Math.max(
                          1,
                          (rowState.page || prev.page || 1) - 1,
                        ),
                      }))
                    }
                    disabled={!rowState.hasPrevious || rowLoading}
                  >
                    Previous
                  </button>
                  <span>Page {rowState.page || rowRequest.page}</span>
                  <button
                    className="secondary"
                    onClick={() =>
                      setRowRequest((prev) => ({
                        ...prev,
                        page: (rowState.page || prev.page || 1) + 1,
                      }))
                    }
                    disabled={!rowState.hasNext || rowLoading}
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="card placeholder-card">
              <p>Select a table to inspect its columns and rows.</p>
            </div>
          )}
        </div>
      </div>
      {actionState.mode === "add" ? (
        <RowEditorModal
          mode="add"
          columns={rowState.columns}
          busy={actionBusy}
          error={actionError}
          onCancel={closeActionModal}
          onSubmit={submitAddRow}
        />
      ) : null}
      {actionState.mode === "edit" ? (
        <RowEditorModal
          mode="edit"
          columns={rowState.columns}
          row={actionState.row}
          busy={actionBusy}
          error={actionError}
          onCancel={closeActionModal}
          onSubmit={submitEditRow}
        />
      ) : null}
      {actionState.mode === "delete" ? (
        <DeleteRowModal
          columns={rowState.columns}
          row={actionState.row}
          busy={actionBusy}
          error={actionError}
          onCancel={closeActionModal}
          onConfirm={confirmDeleteRow}
        />
      ) : null}
    </section>
  );
}

function PolicyDryRunModal({ visible, onClose, namespace, policyNames, apiFetch }) {
  const [sampleText, setSampleText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState(null);

  useEffect(() => {
    if (!visible) {
      setSampleText("");
      setBusy(false);
      setError("");
      setResults(null);
    }
  }, [visible]);

  if (!visible) {
    return null;
  }

  const handleSubmit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResults(null);

    let parsedSamples;
    try {
      const trimmed = sampleText.trim();
      if (!trimmed) {
        throw new Error("Provide sample payloads in JSON format to evaluate.");
      }
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        parsedSamples = parsed.filter(
          (item) => item && typeof item === "object" && !Array.isArray(item),
        );
      } else if (parsed && typeof parsed === "object") {
        parsedSamples = [parsed];
      } else {
        throw new Error("Samples must be a JSON object or array of objects.");
      }
      if (!parsedSamples.length) {
        throw new Error("At least one valid JSON object is required.");
      }
    } catch (parseError) {
      setBusy(false);
      setError(parseError.message || "Unable to parse sample payloads.");
      return;
    }

    try {
      const payload = { samples: parsedSamples };
      if (Array.isArray(policyNames) && policyNames.length) {
        payload.policies = policyNames;
      }
      if (namespace) {
        payload.namespace = namespace;
      }
      const response = await apiFetch("/governance/policies/dry-run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(data.message || "Unable to execute dry run.");
        return;
      }
      setResults(data);
    } catch (submitError) {
      if (submitError.code === "NO_SESSION") {
        setError("Session expired. Sign in again to simulate policies.");
      } else {
        setError("Network error running dry run.");
      }
    } finally {
      setBusy(false);
    }
  };

  const statistics = results?.statistics;
  const reports = Array.isArray(results?.reports) ? results.reports : [];

  return (
    <div className="modal-backdrop">
      <div className="modal modal-wide">
        <h2>Policy dry run</h2>
        <p className="modal-description">
          Paste example memories or payloads to see which rules would trigger before rolling out
          policy updates.
        </p>
        {namespace ? (
          <p className="muted">
            Target namespace: <strong>{namespace}</strong>
          </p>
        ) : null}
        <form className="dry-run-form" onSubmit={handleSubmit}>
          <label className="modal-label">
            Sample payloads (JSON)
            <textarea
              value={sampleText}
              onChange={(event) => setSampleText(event.target.value)}
              placeholder='[{"namespace": "ops/critical", "privacy": 7.5}]'
              rows={8}
              required
              disabled={busy}
            />
          </label>
          {error ? <p className="modal-error">{error}</p> : null}
          <div className="modal-actions">
            <button type="button" className="secondary" onClick={onClose} disabled={busy}>
              Cancel
            </button>
            <button type="submit" className="primary" disabled={busy}>
              {busy ? "Running..." : "Run dry test"}
            </button>
          </div>
        </form>
        {statistics ? (
          <div className="dry-run-results">
            <div className="dry-run-summary">
              <span>
                Policies evaluated: <strong>{statistics.total_policies}</strong>
              </span>
              <span>
                Samples processed: <strong>{statistics.total_samples}</strong>
              </span>
            </div>
            {statistics.violations ? (
              <div className="dry-run-violations">
                <h4>Violation reasons</h4>
                {Object.keys(statistics.violations).length ? (
                  <ul>
                    {Object.entries(statistics.violations).map(([reason, count]) => (
                      <li key={reason}>
                        <strong>{reason}</strong>: {count}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">No violations detected.</p>
                )}
              </div>
            ) : null}
            <div className="dry-run-report-list">
              {reports.map((report) => (
                <div
                  className="dry-run-report"
                  key={report.policy?.name || report.policy_name || report.policy?.id}
                >
                  <h4>{report.policy?.name || "Policy"}</h4>
                  <p className="muted">
                    Action: <strong>{report.policy?.action || "unknown"}</strong> · Triggers: {" "}
                    <strong>{report.trigger_count}</strong>
                  </p>
                  {Array.isArray(report.hits) && report.hits.length ? (
                    <ul>
                      {report.hits.slice(0, 6).map((hit) => (
                        <li key={`${report.policy?.name || "policy"}-${hit.sample_index}`}>
                          Sample #{hit.sample_index}
                          {hit.memory_id ? ` (${hit.memory_id})` : ""} → {hit.reasons.join(", ")}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">No sample triggered this rule.</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function NamespaceSegmentationPanel({ apiFetch, sessionActive }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [segments, setSegments] = useState([]);
  const [filterOptions, setFilterOptions] = useState({
    teams: [],
    workspaces: [],
    privacy_bands: [],
    lifecycle_bands: [],
  });
  const [filters, setFilters] = useState({
    team: "all",
    workspace: "all",
    privacy: "all",
    lifecycle: "all",
  });
  const [selectedNamespace, setSelectedNamespace] = useState("");
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [dryRunContext, setDryRunContext] = useState({
    visible: false,
    namespace: "",
    policies: [],
  });

  const loadSegments = useCallback(async () => {
    if (!sessionActive) {
      setSegments([]);
      setFilterOptions({
        teams: [],
        workspaces: [],
        privacy_bands: [],
        lifecycle_bands: [],
      });
      setSelectedNamespace("");
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (filters.team && filters.team !== "all") {
        params.set("team", filters.team);
      }
      if (filters.workspace && filters.workspace !== "all") {
        params.set("workspace", filters.workspace);
      }
      if (filters.privacy && filters.privacy !== "all") {
        params.set("privacy", filters.privacy);
      }
      if (filters.lifecycle && filters.lifecycle !== "all") {
        params.set("lifecycle", filters.lifecycle);
      }
      const query = params.toString();
      const response = await apiFetch(
        query ? `/governance/namespaces?${query}` : "/governance/namespaces",
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setSegments([]);
        setError(data.message || "Unable to load namespaces");
        return;
      }
      const list = Array.isArray(data.namespaces) ? data.namespaces : [];
      setSegments(list);
      setFilterOptions({
        teams: Array.isArray(data.filters?.teams) ? data.filters.teams : [],
        workspaces: Array.isArray(data.filters?.workspaces) ? data.filters.workspaces : [],
        privacy_bands: Array.isArray(data.filters?.privacy_bands)
          ? data.filters.privacy_bands
          : [],
        lifecycle_bands: Array.isArray(data.filters?.lifecycle_bands)
          ? data.filters.lifecycle_bands
          : [],
      });
      setSelectedNamespace((current) => {
        if (current && list.some((item) => item.namespace === current)) {
          return current;
        }
        return list.length ? list[0].namespace : "";
      });
    } catch (loadError) {
      setSegments([]);
      if (loadError.code !== "NO_SESSION") {
        setError("Unable to load namespaces");
      }
    } finally {
      setLoading(false);
    }
  }, [apiFetch, sessionActive, filters]);

  useEffect(() => {
    loadSegments();
  }, [loadSegments]);

  const fetchDetail = useCallback(
    async (namespace) => {
      if (!sessionActive || !namespace) {
        setDetail(null);
        setDetailError("");
        return;
      }
      setDetailLoading(true);
      setDetailError("");
      try {
        const response = await apiFetch(
          `/governance/namespaces/${encodeURIComponent(namespace)}`,
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          setDetail(null);
          setDetailError(data.message || "Unable to load namespace detail");
          return;
        }
        setDetail(data);
      } catch (detailError) {
        if (detailError.code === "NO_SESSION") {
          setDetailError("Session expired. Sign in again to view policy details.");
        } else {
          setDetailError("Unable to load namespace detail");
        }
        setDetail(null);
      } finally {
        setDetailLoading(false);
      }
    },
    [apiFetch, sessionActive],
  );

  useEffect(() => {
    if (selectedNamespace) {
      fetchDetail(selectedNamespace);
    } else {
      setDetail(null);
      setDetailError("");
    }
  }, [selectedNamespace, fetchDetail]);

  const openDryRun = useCallback(() => {
    if (!detail) {
      return;
    }
    const policyNames = Array.isArray(detail.policies)
      ? detail.policies
          .map((policy) => policy?.name)
          .filter((name) => typeof name === "string" && name)
      : [];
    setDryRunContext({
      visible: true,
      namespace: detail.namespace,
      policies: policyNames,
    });
  }, [detail]);

  const closeDryRun = useCallback(() => {
    setDryRunContext({ visible: false, namespace: "", policies: [] });
  }, []);

  if (!sessionActive) {
    return (
      <section className="section">
        <h2>Namespace segmentation</h2>
        <div className="card placeholder-card">
          <p>Sign in to review governance coverage across namespaces.</p>
        </div>
      </section>
    );
  }

  const selected = segments.find((item) => item.namespace === selectedNamespace) || null;

  return (
    <section className="section">
      <h2>Namespace segmentation</h2>
      <div className="card segmentation-card">
        <div className="card-header">
          <div>
            <h3>Active coverage</h3>
            <p className="card-subtitle">
              Compare rule density, ownership, and escalation bindings across namespaces.
            </p>
          </div>
          <div className="segmentation-filters">
            <label>
              Team
              <select
                value={filters.team}
                onChange={(event) =>
                  setFilters((prev) => ({ ...prev, team: event.target.value }))
                }
              >
                <option value="all">All teams</option>
                {filterOptions.teams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Workspace
              <select
                value={filters.workspace}
                onChange={(event) =>
                  setFilters((prev) => ({ ...prev, workspace: event.target.value }))
                }
              >
                <option value="all">All workspaces</option>
                {filterOptions.workspaces.map((workspace) => (
                  <option key={workspace} value={workspace}>
                    {workspace}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Privacy band
              <select
                value={filters.privacy}
                onChange={(event) =>
                  setFilters((prev) => ({ ...prev, privacy: event.target.value }))
                }
              >
                <option value="all">All</option>
                {filterOptions.privacy_bands.map((band) => (
                  <option key={band} value={band}>
                    {band}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Lifecycle
              <select
                value={filters.lifecycle}
                onChange={(event) =>
                  setFilters((prev) => ({ ...prev, lifecycle: event.target.value }))
                }
              >
                <option value="all">All</option>
                {filterOptions.lifecycle_bands.map((band) => (
                  <option key={band} value={band}>
                    {band}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="segmentation-layout">
          <div className="segmentation-list">
            {loading ? <p className="muted">Loading namespaces…</p> : null}
            {!loading && !segments.length ? (
              <p className="muted">No namespaces found for the selected filters.</p>
            ) : (
              <table className="data-table segmentation-table">
                <thead>
                  <tr>
                    <th scope="col">Namespace</th>
                    <th scope="col">Rule density</th>
                    <th scope="col">Escalations</th>
                    <th scope="col">Owners</th>
                  </tr>
                </thead>
                <tbody>
                  {segments.map((segment) => {
                    const isSelected = segment.namespace === selectedNamespace;
                    const density = Math.round(Math.max(0, Math.min(1, segment.rule_density || 0)) * 100);
                    return (
                      <tr
                        key={segment.namespace}
                        className={isSelected ? "selected" : ""}
                        onClick={() => setSelectedNamespace(segment.namespace)}
                      >
                        <th scope="row">{segment.namespace}</th>
                        <td>
                          <div className="density-bar">
                            <div
                              className="density-bar-fill"
                              style={{ width: `${Math.max(density, 6)}%` }}
                            />
                            <span className="density-label">
                              {segment.rule_count} rule{segment.rule_count === 1 ? "" : "s"}
                            </span>
                          </div>
                        </td>
                        <td>
                          {segment.escalations && segment.escalations.length ? (
                            <span>{segment.escalations.join(", ")}</span>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                        <td>
                          {segment.owners && segment.owners.length ? (
                            <span>{segment.owners.join(", ")}</span>
                          ) : (
                            <span className="muted">Unassigned</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
          <div className="segmentation-detail">
            {detailLoading ? <p className="muted">Loading details…</p> : null}
            {detailError ? <p className="error-text">{detailError}</p> : null}
            {!detail && !detailLoading && !detailError ? (
              <p className="muted">Select a namespace to review policies and controls.</p>
            ) : null}
            {detail ? (
              <div className="detail-card">
                <h3>{detail.namespace}</h3>
                <div className="detail-meta">
                  <span>
                    {selected?.privacy_band ? `Privacy: ${selected.privacy_band}` : ""}
                  </span>
                  <span>
                    {selected?.lifecycle_band ? `Lifecycle: ${selected.lifecycle_band}` : ""}
                  </span>
                  <span>
                    Contacts: {detail.escalations?.length || 0}
                  </span>
                </div>
                <div className="detail-section">
                  <h4>Policies</h4>
                  {Array.isArray(detail.policies) && detail.policies.length ? (
                    <ul>
                      {detail.policies.map((policy) => (
                        <li key={policy.name}>
                          <strong>{policy.name}</strong> → {policy.action}
                          {policy.escalate_to ? ` · escalates to ${policy.escalate_to}` : ""}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">No policies currently target this namespace.</p>
                  )}
                </div>
                <div className="detail-section">
                  <h4>Privacy floors</h4>
                  {detail.privacy_floors && detail.privacy_floors.length ? (
                    <ul>
                      {detail.privacy_floors.map((floor) => (
                        <li key={floor.name}>
                          <strong>{floor.name}</strong> → floor {floor.privacy_floor}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">No namespace-specific privacy floors configured.</p>
                  )}
                </div>
                <div className="detail-section">
                  <h4>Escalation roster</h4>
                  {detail.escalations && detail.escalations.length ? (
                    <ul>
                      {detail.escalations.map((contact) => (
                        <li key={contact.name}>
                          <strong>{contact.name}</strong> via {contact.channel} → {contact.target}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">No escalation contacts tied to this namespace.</p>
                  )}
                </div>
                {detail.metadata_keys && detail.metadata_keys.length ? (
                  <div className="detail-section">
                    <h4>Metadata keys</h4>
                    <p className="muted">{detail.metadata_keys.join(", ")}</p>
                  </div>
                ) : null}
                <div className="detail-actions">
                  <button type="button" className="secondary" onClick={openDryRun}>
                    Test policies with sample dataset
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
      <PolicyDryRunModal
        visible={dryRunContext.visible}
        namespace={dryRunContext.namespace}
        policyNames={dryRunContext.policies}
        onClose={closeDryRun}
        apiFetch={apiFetch}
      />
    </section>
  );
}

function parseDelimitedList(value) {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  return String(value)
    .split(/[,\n]/)
    .map((segment) => segment.trim())
    .filter(Boolean);
}

function buildRosterDraft(contact) {
  if (!contact) {
    return {
      name: "",
      channel: "",
      target: "",
      priority: "",
      namespaces: "",
      triggers: "",
      coverage: "",
      rotation: "",
      channels: "",
      notes: "",
      syncProvider: "",
      syncStatus: "",
      integrations: "",
    };
  }
  const metadata = contact.metadata || {};
  const rotationText = Array.isArray(metadata.rotation)
    ? metadata.rotation
        .map((entry) => {
          if (!entry || typeof entry !== "object") {
            return null;
          }
          const date = entry.date || "";
          const primary = entry.primary || "";
          const secondary = entry.secondary || "";
          if (!date && !primary && !secondary) {
            return null;
          }
          return [date, primary, secondary].filter(Boolean).join(" | ");
        })
        .filter(Boolean)
        .join("\n")
    : "";
  return {
    name: contact.name || "",
    channel: contact.channel || "",
    target: contact.target || "",
    priority: contact.priority || "",
    namespaces: Array.isArray(metadata.namespaces)
      ? metadata.namespaces.join(", ")
      : typeof metadata.namespaces === "string"
      ? metadata.namespaces
      : "",
    triggers: Array.isArray(metadata.triggers)
      ? metadata.triggers.join(", ")
      : typeof metadata.triggers === "string"
      ? metadata.triggers
      : "",
    coverage: metadata.coverage || "",
    rotation: rotationText,
    channels: Array.isArray(metadata.channels)
      ? metadata.channels.join(", ")
      : typeof metadata.channels === "string"
      ? metadata.channels
      : "",
    integrations: Array.isArray(metadata.integrations)
      ? metadata.integrations
          .map((integration) => {
            if (!integration || typeof integration !== "object") {
              return null;
            }
            const type = integration.type || "";
            const target = integration.target || "";
            const status = integration.status || "";
            const parts = [type, target, status].map((part) => String(part || "").trim());
            if (!parts.some(Boolean)) {
              return null;
            }
            return parts.filter(Boolean).join(" | ");
          })
          .filter(Boolean)
          .join("\n")
      : typeof metadata.integrations === "string"
      ? metadata.integrations
      : "",
    notes: metadata.notes || "",
    syncProvider: metadata.sync?.provider || "",
    syncStatus: metadata.sync?.status || "",
  };
}

function buildContactPayload(draft) {
  const metadata = {};
  const namespaces = parseDelimitedList(draft.namespaces);
  if (namespaces.length) {
    metadata.namespaces = namespaces;
  }
  const triggers = parseDelimitedList(draft.triggers);
  if (triggers.length) {
    metadata.triggers = triggers;
  }
  if (draft.coverage && draft.coverage.trim()) {
    metadata.coverage = draft.coverage.trim();
  }
  const channels = parseDelimitedList(draft.channels);
  if (channels.length) {
    metadata.channels = channels;
  }
  if (draft.notes && draft.notes.trim()) {
    metadata.notes = draft.notes.trim();
  }
  const integrationLines = String(draft.integrations || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (integrationLines.length) {
    const integrations = integrationLines
      .map((line) => {
        const segments = line.split("|").map((segment) => segment.trim());
        if (!segments.some(Boolean)) {
          return null;
        }
        const [type, target, status] = segments;
        const entry = {};
        if (type) {
          entry.type = type;
        }
        if (target) {
          entry.target = target;
        }
        if (status) {
          entry.status = status.toLowerCase();
        }
        return Object.keys(entry).length ? entry : null;
      })
      .filter(Boolean);
    if (integrations.length) {
      metadata.integrations = integrations;
    }
  }
  const rotationLines = String(draft.rotation || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (rotationLines.length) {
    const rotation = rotationLines
      .map((line) => {
        const parts = line.split("|").map((part) => part.trim()).filter(Boolean);
        if (!parts.length) {
          return null;
        }
        const entry = {};
        if (parts[0]) {
          entry.date = parts[0];
        }
        if (parts[1]) {
          entry.primary = parts[1];
        }
        if (parts[2]) {
          entry.secondary = parts[2];
        }
        return Object.keys(entry).length ? entry : null;
      })
      .filter(Boolean);
    if (rotation.length) {
      metadata.rotation = rotation;
    }
  }
  if (draft.syncProvider && draft.syncProvider.trim()) {
    metadata.sync = {
      provider: draft.syncProvider.trim(),
      status: draft.syncStatus && draft.syncStatus.trim() ? draft.syncStatus.trim() : "active",
    };
  } else if (draft.syncStatus && draft.syncStatus.trim()) {
    metadata.sync = { status: draft.syncStatus.trim() };
  }
  const payload = {
    name: (draft.name || "").trim(),
    channel: (draft.channel || "").trim(),
    target: (draft.target || "").trim(),
    metadata,
  };
  if (draft.priority && draft.priority.trim()) {
    payload.priority = draft.priority.trim();
  }
  return payload;
}

function RosterEditorModal({
  visible,
  mode,
  draft,
  onChange,
  onClose,
  onSubmit,
  busy,
  error,
}) {
  if (!visible) {
    return null;
  }

  const handleChange = (event) => {
    const { name, value } = event.target;
    onChange(name, value);
  };

  return (
    <div className="modal-backdrop">
      <div className="modal modal-wide">
        <h2>{mode === "edit" ? "Update escalation queue" : "Create escalation queue"}</h2>
        <p className="modal-description">
          Define escalation coverage, namespace bindings, and integration metadata.
        </p>
        <form
          className="roster-form"
          onSubmit={(event) => {
            event.preventDefault();
            onSubmit();
          }}
        >
          <div className="form-grid">
            <label>
              Name
              <input
                name="name"
                value={draft.name}
                onChange={handleChange}
                required
                disabled={busy}
                placeholder="SecOps (primary)"
              />
            </label>
            <label>
              Channel
              <input
                name="channel"
                value={draft.channel}
                onChange={handleChange}
                required
                disabled={busy}
                placeholder="slack / pagerduty / email"
              />
            </label>
            <label>
              Target
              <input
                name="target"
                value={draft.target}
                onChange={handleChange}
                required
                disabled={busy}
                placeholder="pagerduty:service-id or #channel"
              />
            </label>
            <label>
              Priority
              <select
                name="priority"
                value={draft.priority}
                onChange={handleChange}
                disabled={busy}
              >
                <option value="">Default</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </label>
            <label>
              Namespaces
              <input
                name="namespaces"
                value={draft.namespaces}
                onChange={handleChange}
                disabled={busy}
                placeholder="ops/*, partners/high"
              />
            </label>
            <label>
              Trigger sources
              <input
                name="triggers"
                value={draft.triggers}
                onChange={handleChange}
                disabled={busy}
                placeholder="privacy_ceiling, lifecycle"
              />
            </label>
            <label>
              Coverage window
              <input
                name="coverage"
                value={draft.coverage}
                onChange={handleChange}
                disabled={busy}
                placeholder="Week 42 · 24/7"
              />
            </label>
            <label>
              Additional channels
              <input
                name="channels"
                value={draft.channels}
                onChange={handleChange}
                disabled={busy}
                placeholder="slack:#sec-urgent, email:oncall@example.com"
              />
            </label>
            <label>
              Rotation (one per line — date | primary | secondary)
              <textarea
                name="rotation"
                value={draft.rotation}
                onChange={handleChange}
                disabled={busy}
                rows={4}
              />
            </label>
            <label>
              Notification integrations (type | target | status)
              <textarea
                name="integrations"
                value={draft.integrations}
                onChange={handleChange}
                disabled={busy}
                rows={3}
                placeholder="pagerduty | service-id | synced"
              />
            </label>
            <label>
              Notes
              <textarea
                name="notes"
                value={draft.notes}
                onChange={handleChange}
                disabled={busy}
                rows={3}
              />
            </label>
            <label>
              Sync provider
              <input
                name="syncProvider"
                value={draft.syncProvider}
                onChange={handleChange}
                disabled={busy}
                placeholder="pagerduty"
              />
            </label>
            <label>
              Sync status
              <input
                name="syncStatus"
                value={draft.syncStatus}
                onChange={handleChange}
                disabled={busy}
                placeholder="synced / manual"
              />
            </label>
          </div>
          {error ? <p className="modal-error">{error}</p> : null}
          <div className="modal-actions">
            <button type="button" className="secondary" onClick={onClose} disabled={busy}>
              Cancel
            </button>
            <button type="submit" className="primary" disabled={busy}>
              {busy ? "Saving..." : mode === "edit" ? "Save changes" : "Create queue"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function EscalationRosterPanel({ apiFetch, sessionActive, notify }) {
  const [contacts, setContacts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const [previewMessage, setPreviewMessage] = useState("");
  const [previewStatus, setPreviewStatus] = useState("");
  const [rosterBusy, setRosterBusy] = useState(false);
  const [verification, setVerification] = useState(null);
  const [verificationLoading, setVerificationLoading] = useState(false);
  const [verificationError, setVerificationError] = useState("");
  const [rotationStatus, setRotationStatus] = useState(null);
  const [rotationLoading, setRotationLoading] = useState(false);
  const [rotationError, setRotationError] = useState("");
  const [editorState, setEditorState] = useState({
    visible: false,
    mode: "add",
    draft: buildRosterDraft(null),
    originalName: null,
    busy: false,
    error: "",
  });

  const loadContacts = useCallback(async () => {
    if (!sessionActive) {
      setContacts([]);
      setSelectedName("");
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch("/governance/escalations");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setContacts([]);
        setError(data.message || "Unable to load escalation rosters");
        return;
      }
      const list = Array.isArray(data.contacts) ? data.contacts : [];
      setContacts(list);
      setSelectedName((current) => {
        if (current && list.some((item) => item.name === current)) {
          return current;
        }
        return list.length ? list[0].name : "";
      });
    } catch (loadError) {
      setContacts([]);
      if (loadError.code !== "NO_SESSION") {
        setError("Unable to load escalation rosters");
      }
    } finally {
      setLoading(false);
    }
  }, [apiFetch, sessionActive]);

  useEffect(() => {
    loadContacts();
  }, [loadContacts]);

  const loadVerification = useCallback(
    async (refresh = false) => {
      if (!sessionActive) {
        setVerification(null);
        setVerificationError("");
        return;
      }
      setVerificationLoading(true);
      setVerificationError("");
      try {
        const params = new URLSearchParams({ cadence: "60" });
        if (refresh) {
          params.set("refresh", "1");
        }
        const response = await apiFetch(
          `/governance/escalations/verification?${params.toString()}`,
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          setVerification(null);
          setVerificationError(data.message || "Unable to load roster verification");
          return;
        }
        setVerification(data.verification || null);
      } catch (verificationError) {
        if (verificationError.code !== "NO_SESSION") {
          setVerificationError("Unable to load roster verification");
        }
        setVerification(null);
      } finally {
        setVerificationLoading(false);
      }
    },
    [apiFetch, sessionActive],
  );

  useEffect(() => {
    loadVerification(false);
  }, [loadVerification]);

  const refreshVerification = useCallback(() => {
    loadVerification(true);
  }, [loadVerification]);

  const loadRotation = useCallback(
    async (refresh = false) => {
      if (!sessionActive) {
        setRotationStatus(null);
        setRotationError("");
        return;
      }
      setRotationLoading(true);
      setRotationError("");
      try {
        const params = new URLSearchParams({ cadence: "60" });
        if (refresh) {
          params.set("refresh", "1");
        }
        const response = await apiFetch(
          `/governance/escalations/rotation?${params.toString()}`,
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          setRotationStatus(null);
          setRotationError(data.message || "Unable to load rotation status");
          return;
        }
        setRotationStatus(data.rotation || null);
      } catch (rotationErr) {
        if (rotationErr.code !== "NO_SESSION") {
          setRotationError("Unable to load rotation status");
        }
        setRotationStatus(null);
      } finally {
        setRotationLoading(false);
      }
    },
    [apiFetch, sessionActive],
  );

  useEffect(() => {
    loadRotation(false);
  }, [loadRotation]);

  const refreshRotation = useCallback(() => {
    loadRotation(true);
  }, [loadRotation]);

  const selectedContact = useMemo(
    () => contacts.find((contact) => contact.name === selectedName) || null,
    [contacts, selectedName],
  );

  const selectedVerification = useMemo(() => {
    if (!verification || !selectedContact || !Array.isArray(verification.contacts)) {
      return null;
    }
    return (
      verification.contacts.find((item) => item.name === selectedContact.name) || null
    );
  }, [selectedContact, verification]);

  const selectedRotation = useMemo(() => {
    if (!rotationStatus || !selectedContact || !Array.isArray(rotationStatus.contacts)) {
      return null;
    }
    return (
      rotationStatus.contacts.find((item) => item.name === selectedContact.name) || null
    );
  }, [rotationStatus, selectedContact]);

  const openCreate = useCallback(() => {
    setEditorState({
      visible: true,
      mode: "add",
      draft: buildRosterDraft(null),
      originalName: null,
      busy: false,
      error: "",
    });
  }, []);

  const openEdit = useCallback(() => {
    if (!selectedContact) {
      return;
    }
    setEditorState({
      visible: true,
      mode: "edit",
      draft: buildRosterDraft(selectedContact),
      originalName: selectedContact.name,
      busy: false,
      error: "",
    });
  }, [selectedContact]);

  const closeEditor = useCallback(() => {
    setEditorState((prev) => {
      if (prev.busy) {
        return prev;
      }
      return { ...prev, visible: false, error: "" };
    });
  }, []);

  const updateDraft = useCallback((name, value) => {
    setEditorState((prev) => ({
      ...prev,
      draft: { ...prev.draft, [name]: value },
    }));
  }, []);

  const saveRoster = useCallback(async () => {
    const payload = buildContactPayload(editorState.draft);
    if (!payload.name || !payload.channel || !payload.target) {
      setEditorState((prev) => ({
        ...prev,
        error: "Name, channel, and target are required.",
      }));
      return;
    }
    setEditorState((prev) => ({ ...prev, busy: true, error: "" }));
    try {
      const method = editorState.mode === "edit" ? "PUT" : "POST";
      const url =
        editorState.mode === "edit"
          ? `/governance/escalations/${encodeURIComponent(
              editorState.originalName || payload.name,
            )}`
          : "/governance/escalations";
      const response = await apiFetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contact: payload }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setEditorState((prev) => ({
          ...prev,
          busy: false,
          error: data.message || "Unable to save roster.",
        }));
        return;
      }
      await loadContacts();
      setEditorState({
        visible: false,
        mode: "add",
        draft: buildRosterDraft(null),
        originalName: null,
        busy: false,
        error: "",
      });
      if (notify) {
        notify({
          type: "success",
          message:
            editorState.mode === "edit"
              ? "Escalation queue updated"
              : "Escalation queue created",
        });
      }
    } catch (saveError) {
      if (saveError.code === "NO_SESSION") {
        setEditorState((prev) => ({
          ...prev,
          busy: false,
          error: "Session expired. Sign in again to save changes.",
        }));
      } else {
        setEditorState((prev) => ({
          ...prev,
          busy: false,
          error: "Unable to save roster.",
        }));
      }
    }
  }, [apiFetch, editorState, loadContacts, notify]);

  const deleteRoster = useCallback(async () => {
    if (!selectedContact || rosterBusy) {
      return;
    }
    setRosterBusy(true);
    setPreviewStatus("");
    try {
      const response = await apiFetch(
        `/governance/escalations/${encodeURIComponent(selectedContact.name)}`,
        { method: "DELETE" },
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(data.message || "Unable to delete roster");
        return;
      }
      await loadContacts();
      setPreviewMessage("");
      setPreviewStatus("");
      if (notify) {
        notify({ type: "success", message: "Escalation queue removed" });
      }
    } catch (deleteError) {
      if (deleteError.code === "NO_SESSION") {
        setError("Session expired. Sign in again to manage rosters.");
      } else {
        setError("Unable to delete roster");
      }
    } finally {
      setRosterBusy(false);
    }
  }, [apiFetch, loadContacts, notify, rosterBusy, selectedContact]);

  const previewNotification = useCallback(async () => {
    if (!selectedContact) {
      return;
    }
    setRosterBusy(true);
    setPreviewStatus("");
    try {
      const response = await apiFetch(
        `/governance/escalations/${encodeURIComponent(selectedContact.name)}/preview`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: previewMessage }),
        },
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setPreviewStatus(data.message || "Unable to preview notification.");
        return;
      }
      if (data.preview) {
        setPreviewStatus(
          `Preview ready → ${data.preview.channel} → ${data.preview.target}`,
        );
      } else {
        setPreviewStatus("Preview dispatched.");
      }
      if (notify) {
        notify({ type: "success", message: "Preview notification simulated" });
      }
    } catch (previewError) {
      if (previewError.code === "NO_SESSION") {
        setPreviewStatus("Session expired. Sign in again to preview notifications.");
      } else {
        setPreviewStatus("Unable to preview notification.");
      }
    } finally {
      setRosterBusy(false);
    }
  }, [apiFetch, notify, previewMessage, selectedContact]);

  if (!sessionActive) {
    return (
      <section className="section">
        <h2>Escalation rosters</h2>
        <div className="card placeholder-card">
          <p>Sign in to manage on-call queues and escalation contacts.</p>
        </div>
      </section>
    );
  }

  const metadata = (selectedContact && selectedContact.metadata) || {};
  const namespaces = parseDelimitedList(metadata.namespaces);
  const triggers = parseDelimitedList(metadata.triggers);
  const rotationEntries = Array.isArray(metadata.rotation) ? metadata.rotation : [];
  const channels = parseDelimitedList(metadata.channels);
  const syncInfo = metadata.sync || {};
  const integrationEntries = useMemo(() => {
    if (selectedVerification && Array.isArray(selectedVerification.integrations)) {
      return selectedVerification.integrations;
    }
    if (Array.isArray(metadata.integrations)) {
      return metadata.integrations;
    }
    return [];
  }, [metadata.integrations, selectedVerification]);

  return (
    <section className="section">
      <h2>Escalation rosters</h2>
      <div className="card roster-card">
        <div className="card-header">
          <div>
            <h3>Coverage queues</h3>
            <p className="card-subtitle">
              Map namespaces to incident queues and confirm external integrations.
            </p>
          </div>
          <div className="roster-actions">
            <button type="button" className="primary" onClick={openCreate}>
              Add queue
            </button>
          </div>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="roster-verification">
          <div className="verification-header">
            <h4>Roster verification</h4>
            <button
              type="button"
              className="secondary"
              onClick={refreshVerification}
              disabled={verificationLoading || rosterBusy}
            >
              {verificationLoading ? "Checking…" : "Run check"}
            </button>
          </div>
          {verificationError ? <p className="error-text">{verificationError}</p> : null}
          {!verification && !verificationLoading && !verificationError ? (
            <p className="muted">No verification runs recorded yet.</p>
          ) : null}
          {verification ? (
            <div className="verification-summary">
              <p className="muted">
                Last run: {formatAuditTimestamp(verification.generated_at)}
              </p>
              <p className="muted">
                Next check: {formatAuditTimestamp(verification.next_check_at)}
              </p>
              <div className="chip-row">
                {Object.entries(verification.summary?.status_counts || {}).map(
                  ([status, count]) => (
                    <span key={status} className={`chip status-${status}`}>
                      {status}: {count}
                    </span>
                  ),
                )}
                {typeof verification.summary?.total_issues === "number" ? (
                  <span className="chip muted">
                    Issues: {verification.summary.total_issues}
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>
        <div className="roster-automation">
          <div className="verification-header">
            <h4>Rotation automation</h4>
            <button
              type="button"
              className="secondary"
              onClick={refreshRotation}
              disabled={rotationLoading || rosterBusy}
            >
              {rotationLoading ? "Syncing…" : "Refresh"}
            </button>
          </div>
          {rotationError ? <p className="error-text">{rotationError}</p> : null}
          {!rotationStatus && !rotationLoading && !rotationError ? (
            <p className="muted">No rotation runs recorded yet.</p>
          ) : null}
          {rotationStatus ? (
            <div className="verification-summary">
              <p className="muted">
                Last run: {formatAuditTimestamp(rotationStatus.generated_at)}
              </p>
              <p className="muted">
                Next check: {formatAuditTimestamp(rotationStatus.next_check_at)}
              </p>
              <div className="chip-row">
                <span className="chip muted">
                  Updates: {rotationStatus.summary?.metadata_updates ?? 0}
                </span>
                <span className="chip muted">
                  Overdue: {rotationStatus.summary?.overdue_contacts ?? 0}
                </span>
                <span className="chip muted">
                  Queues: {rotationStatus.summary?.total_contacts ?? 0}
                </span>
              </div>
            </div>
          ) : null}
        </div>
        <div className="roster-layout">
          <div className="roster-list">
            {loading ? <p className="muted">Loading queues…</p> : null}
            {!loading && !contacts.length ? (
              <p className="muted">No escalation queues configured yet.</p>
            ) : (
              <ul>
                {contacts.map((contact) => (
                  <li
                    key={contact.name}
                    className={contact.name === selectedName ? "selected" : ""}
                  >
                    <button type="button" onClick={() => setSelectedName(contact.name)}>
                      <span className="roster-name">{contact.name}</span>
                      <span className="roster-meta">
                        {contact.metadata?.coverage || "Coverage unknown"}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="roster-detail">
            {selectedContact ? (
              <div className="detail-card">
                <h3>{selectedContact.name}</h3>
                <div className="detail-meta">
                  <span>{selectedContact.channel}</span>
                  <span>{selectedContact.target}</span>
                  {selectedContact.priority ? (
                    <span>Priority: {selectedContact.priority}</span>
                  ) : null}
                </div>
                <div className="detail-section">
                  <h4>Bound namespaces</h4>
                  {namespaces.length ? (
                    <p className="muted">{namespaces.join(", ")}</p>
                  ) : (
                    <p className="muted">All namespaces ("*")</p>
                  )}
                </div>
                <div className="detail-section">
                  <h4>Trigger sources</h4>
                  {triggers.length ? (
                    <p className="muted">{triggers.join(", ")}</p>
                  ) : (
                    <p className="muted">No trigger metadata defined.</p>
                  )}
                </div>
                <div className="detail-section">
                  <h4>Coverage</h4>
                  <p className="muted">{metadata.coverage || "Coverage not specified."}</p>
                </div>
                {selectedVerification ? (
                  <div className={`detail-section verification-${selectedVerification.status}`}>
                    <h4>Verification status</h4>
                    <p className={`status-pill ${selectedVerification.status}`}>
                      {selectedVerification.status || "unknown"}
                    </p>
                    {selectedVerification.next_rotation ? (
                      <p className="muted">
                        Next rotation: {formatAuditTimestamp(selectedVerification.next_rotation)}
                      </p>
                    ) : null}
                    {Array.isArray(selectedVerification.issues) && selectedVerification.issues.length ? (
                      <ul className="issue-list">
                        {selectedVerification.issues.map((issue, index) => (
                          <li key={`issue-${index}`}>{issue}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="muted">No outstanding verification issues.</p>
                    )}
                  </div>
                ) : null}
                {selectedRotation ? (
                  <div
                    className={`detail-section rotation-${
                      selectedRotation.overdue_windows ? "warning" : "ok"
                    }`}
                  >
                    <h4>Rotation tracker</h4>
                    {selectedRotation.active_rotation ? (
                      <p className="muted">
                        Active since {formatAuditTimestamp(selectedRotation.active_rotation.date)} ·
                        Primary: {selectedRotation.active_rotation.primary || "—"} · Secondary: {" "}
                        {selectedRotation.active_rotation.secondary || "—"}
                      </p>
                    ) : (
                      <p className="muted">No active rotation captured.</p>
                    )}
                    {selectedRotation.next_rotation ? (
                      <p className="muted">
                        Next change: {formatAuditTimestamp(selectedRotation.next_rotation.date)} → {" "}
                        {selectedRotation.next_rotation.primary || "—"}
                      </p>
                    ) : (
                      <p className="muted">No upcoming rotation window documented.</p>
                    )}
                    {selectedRotation.overdue_windows ? (
                      <p className="error-text">
                        {selectedRotation.overdue_windows} rotation window(s) pending updates.
                      </p>
                    ) : null}
                  </div>
                ) : null}
                <div className="detail-section">
                  <h4>Escalation channels</h4>
                  {channels.length ? (
                    <ul>
                      {channels.map((channel) => (
                        <li key={channel}>{channel}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">No secondary channels listed.</p>
                  )}
                </div>
                <div className="detail-section">
                  <h4>Upcoming coverage</h4>
                  {rotationEntries.length ? (
                    <table className="data-table rotation-table">
                      <thead>
                        <tr>
                          <th scope="col">Date</th>
                          <th scope="col">Primary</th>
                          <th scope="col">Secondary</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rotationEntries.map((entry, index) => (
                          <tr key={`${selectedContact.name}-rotation-${index}`}>
                            <td>{entry.date || "—"}</td>
                            <td>{entry.primary || "—"}</td>
                            <td>{entry.secondary || "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p className="muted">No rotation schedule captured.</p>
                  )}
                </div>
                <div className="detail-section">
                  <h4>Integration sync</h4>
                  <p className="muted">
                    {syncInfo.provider
                      ? `${syncInfo.provider} · ${syncInfo.status || "active"}`
                      : syncInfo.status
                      ? syncInfo.status
                      : "No sync metadata"}
                  </p>
                </div>
                <div className="detail-section">
                  <h4>Notification integrations</h4>
                  {integrationEntries.length ? (
                    <ul>
                      {integrationEntries.map((entry, index) => {
                        const type = entry.type || entry.name || "integration";
                        const target = entry.target || entry.endpoint || "";
                        const statusLabel = entry.status || entry.state || "unknown";
                        return (
                          <li key={`integration-${index}`}>
                            <span className="integration-label">
                              {[type, target].filter(Boolean).join(" → ") || type}
                            </span>
                            {statusLabel ? (
                              <span className={`status-pill small ${statusLabel}`}>
                                {statusLabel}
                              </span>
                            ) : null}
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="muted">No integrations documented.</p>
                  )}
                </div>
                {metadata.notes ? (
                  <div className="detail-section">
                    <h4>Notes</h4>
                    <p className="muted">{metadata.notes}</p>
                  </div>
                ) : null}
                <div className="detail-section">
                  <label className="preview-input">
                    Preview message
                    <textarea
                      value={previewMessage}
                      onChange={(event) => setPreviewMessage(event.target.value)}
                      rows={3}
                      disabled={rosterBusy}
                      placeholder="Test payload to send with the preview notification"
                    />
                  </label>
                </div>
                <div className="detail-actions">
                  <button
                    type="button"
                    className="secondary"
                    onClick={openEdit}
                    disabled={editorState.visible || rosterBusy}
                  >
                    Edit queue
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    onClick={previewNotification}
                    disabled={rosterBusy}
                  >
                    Preview notification
                  </button>
                  <button
                    type="button"
                    className="danger"
                    onClick={deleteRoster}
                    disabled={rosterBusy}
                  >
                    Remove queue
                  </button>
                </div>
                {previewStatus ? <p className="status-text">{previewStatus}</p> : null}
              </div>
            ) : (
              <p className="muted">Select a queue to view coverage and escalation metadata.</p>
            )}
          </div>
        </div>
      </div>
      <RosterEditorModal
        visible={editorState.visible}
        mode={editorState.mode}
        draft={editorState.draft}
        onChange={updateDraft}
        onClose={closeEditor}
        onSubmit={saveRoster}
        busy={editorState.busy}
        error={editorState.error}
      />
    </section>
  );
}

function EnforcementTelemetryPanel({ apiFetch, sessionActive }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadTelemetry = useCallback(async () => {
    if (!sessionActive) {
      setData(null);
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch("/governance/telemetry?limit=40");
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(payload.message || "Unable to load telemetry");
        setData(null);
        return;
      }
      setData(payload);
    } catch (telemetryError) {
      if (telemetryError.code !== "NO_SESSION") {
        setError("Unable to load telemetry");
      }
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [apiFetch, sessionActive]);

  useEffect(() => {
    loadTelemetry();
  }, [loadTelemetry]);

  const refresh = useCallback(() => {
    loadTelemetry();
  }, [loadTelemetry]);

  const metrics = Array.isArray(data?.policy_actions) ? data.policy_actions : [];
  const stageCounts = data?.stage_counts || {};

  if (!sessionActive) {
    return null;
  }

  return (
    <section className="section">
      <h2>Enforcement telemetry</h2>
      <div className="card">
        <div className="card-header">
          <div>
            <h3>Policy action trends</h3>
            <p className="card-subtitle">
              Monitor retention outcomes and runtime guardrails.
            </p>
          </div>
          <div className="roster-actions">
            <button type="button" className="secondary" onClick={refresh} disabled={loading}>
              {loading ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        {loading ? <p className="muted">Loading telemetry…</p> : null}
        {!loading && !metrics.length ? (
          <p className="muted">No enforcement activity recorded yet.</p>
        ) : null}
        {!loading && metrics.length ? (
          <div className="telemetry-grid">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Policy</th>
                  <th scope="col">Action</th>
                  <th scope="col">Count</th>
                  <th scope="col">Avg (ms)</th>
                  <th scope="col">Max (ms)</th>
                  <th scope="col">Last triggered</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((item, index) => (
                  <tr key={`${item.policy}-${item.action}-${index}`}>
                    <td>{item.policy}</td>
                    <td>{item.action}</td>
                    <td>{item.count}</td>
                    <td>
                      {item.average_duration_ms != null
                        ? Number(item.average_duration_ms).toFixed(2)
                        : "—"}
                    </td>
                    <td>
                      {item.max_duration_ms != null
                        ? Number(item.max_duration_ms).toFixed(2)
                        : "—"}
                    </td>
                    <td>{formatAuditTimestamp(item.last_triggered_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {Object.keys(stageCounts).length ? (
              <div className="detail-section">
                <h4>Stage totals</h4>
                <div className="chip-row">
                  {Object.entries(stageCounts).map(([stage, actions]) => (
                    <span key={stage} className="chip">
                      {stage}: {Object.values(actions || {}).reduce((sum, value) => sum + value, 0)}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function PolicyOverridePanel({ apiFetch, sessionActive, notify }) {
  const [overrides, setOverrides] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [windowDays, setWindowDays] = useState(14);
  const [automation, setAutomation] = useState(null);
  const [automationLoading, setAutomationLoading] = useState(false);
  const [automationError, setAutomationError] = useState("");

  const loadOverrides = useCallback(async () => {
    if (!sessionActive) {
      setOverrides([]);
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch(`/governance/overrides?window=${windowDays}`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        setOverrides([]);
        setError(payload.message || "Unable to load overrides");
        return;
      }
      setOverrides(Array.isArray(payload.overrides) ? payload.overrides : []);
    } catch (overrideError) {
      if (overrideError.code !== "NO_SESSION") {
        setError("Unable to load overrides");
      }
      setOverrides([]);
    } finally {
      setLoading(false);
    }
  }, [apiFetch, sessionActive, windowDays]);

  useEffect(() => {
    loadOverrides();
  }, [loadOverrides]);

  const loadAutomation = useCallback(
    async (refresh = false) => {
      if (!sessionActive) {
        setAutomation(null);
        setAutomationError("");
        return;
      }
      setAutomationLoading(true);
      setAutomationError("");
      try {
        const params = new URLSearchParams({ cadence: "60" });
        if (refresh) {
          params.set("refresh", "1");
        }
        const response = await apiFetch(
          `/governance/overrides/automation?${params.toString()}`,
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          setAutomation(null);
          setAutomationError(payload.message || "Unable to load automation status");
          return;
        }
        setAutomation(payload.automation || null);
      } catch (automationErr) {
        if (automationErr.code !== "NO_SESSION") {
          setAutomationError("Unable to load automation status");
        }
        setAutomation(null);
      } finally {
        setAutomationLoading(false);
      }
    },
    [apiFetch, sessionActive],
  );

  useEffect(() => {
    loadAutomation(false);
  }, [loadAutomation]);

  const expireOverride = useCallback(
    async (name) => {
      try {
        const response = await apiFetch(
          `/governance/overrides/${encodeURIComponent(name)}/expire`,
          { method: "POST" },
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          if (payload?.message) {
            setError(payload.message);
          }
          return;
        }
        if (notify) {
          notify({ type: "success", message: `Override '${name}' expired` });
        }
        loadOverrides();
        loadAutomation();
      } catch (expireError) {
        if (expireError.code !== "NO_SESSION") {
          setError("Unable to expire override");
        }
      }
    },
    [apiFetch, loadAutomation, loadOverrides, notify],
  );

  const handleWindowChange = useCallback((event) => {
    const value = Number.parseInt(event.target.value, 10);
    if (!Number.isNaN(value) && value > 0) {
      setWindowDays(Math.min(Math.max(value, 1), 365));
    }
  }, []);

  const refreshOverrides = useCallback(() => {
    loadOverrides();
    loadAutomation(true);
  }, [loadAutomation, loadOverrides]);

  if (!sessionActive) {
    return null;
  }

  return (
    <section className="section">
      <h2>Policy overrides</h2>
      <div className="card">
        <div className="card-header">
          <div>
            <h3>Override lifecycle</h3>
            <p className="card-subtitle">
              Highlight temporary exceptions and approaching expirations.
            </p>
          </div>
          <div className="roster-actions">
            <label className="inline-input">
              Window (days)
              <input
                type="number"
                min="1"
                max="365"
                value={windowDays}
                onChange={handleWindowChange}
                disabled={loading}
              />
            </label>
            <button
              type="button"
              className="secondary"
              onClick={refreshOverrides}
              disabled={loading || automationLoading}
            >
              {loading ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        {automationError ? <p className="error-text">{automationError}</p> : null}
        {!automation && !automationLoading && !automationError ? (
          <p className="muted">No automation checks have run yet.</p>
        ) : null}
        {automation ? (
          <div className="verification-summary">
            <p className="muted">Last automation run: {formatAuditTimestamp(automation.generated_at)}</p>
            <p className="muted">Next sweep: {formatAuditTimestamp(automation.next_check_at)}</p>
            <div className="chip-row">
              <span className="chip muted">
                Overrides: {automation.summary?.total_overrides ?? overrides.length}
              </span>
              <span className="chip muted">
                Auto-expired: {automation.summary?.expired_overrides ?? 0}
              </span>
              <span className="chip muted">
                Metadata updates: {automation.summary?.metadata_updates ?? 0}
              </span>
            </div>
          </div>
        ) : null}
        {loading ? <p className="muted">Loading overrides…</p> : null}
        {!loading && !overrides.length ? (
          <p className="muted">No overrides configured.</p>
        ) : null}
        {!loading && overrides.length ? (
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Name</th>
                <th scope="col">Status</th>
                <th scope="col">Expires</th>
                <th scope="col">Action</th>
                <th scope="col">Namespaces</th>
                <th scope="col">Policies</th>
                <th scope="col">Controls</th>
              </tr>
            </thead>
            <tbody>
              {overrides.map((override) => (
                <tr key={override.name}>
                  <td>{override.name}</td>
                  <td>
                    <span className={`status-pill ${override.status || "unknown"}`}>
                      {override.status || "unknown"}
                    </span>
                  </td>
                  <td>{formatAuditTimestamp(override.expires_at)}</td>
                  <td>{override.action}</td>
                  <td>{override.namespaces?.join(", ") || "*"}</td>
                  <td>{override.target_policies?.join(", ") || "—"}</td>
                  <td>
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => expireOverride(override.name)}
                      disabled={override.status === "expired"}
                    >
                      Expire now
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </section>
  );
}

function formatAuditTimestamp(value) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function PolicyAuditExplorer({ apiFetch, sessionActive, notify }) {
  const [filters, setFilters] = useState({
    action: "all",
    namespace: "",
    escalation: "all",
    role: "all",
  });
  const [filterOptions, setFilterOptions] = useState({
    actions: [],
    escalations: [],
    roles: [],
  });
  const [audits, setAudits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [total, setTotal] = useState(0);
  const [selectedAuditId, setSelectedAuditId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");

  const resetDetail = useCallback(() => {
    setDetail(null);
    setDetailError("");
  }, []);

  const loadAudits = useCallback(async () => {
    if (!sessionActive) {
      setAudits([]);
      setFilterOptions({ actions: [], escalations: [], roles: [] });
      setTotal(0);
      setSelectedAuditId(null);
      resetDetail();
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      params.set("page", String(page));
      params.set("page_size", String(pageSize));
      if (filters.action && filters.action !== "all") {
        params.set("action", filters.action);
      }
      if (filters.namespace) {
        params.set("namespace", filters.namespace);
      }
      if (filters.escalation && filters.escalation !== "all") {
        params.set("escalation", filters.escalation);
      }
      if (filters.role && filters.role !== "all") {
        params.set("role", filters.role);
      }
      const response = await apiFetch(`/governance/audits?${params.toString()}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setAudits([]);
        setError(data.message || "Unable to load audit events");
        return;
      }
      const list = Array.isArray(data.audits) ? data.audits : [];
      setAudits(list);
      setTotal(typeof data.total === "number" ? data.total : list.length);
      setFilterOptions({
        actions: Array.isArray(data.filters?.actions) ? data.filters.actions : [],
        escalations: Array.isArray(data.filters?.escalations)
          ? data.filters.escalations
          : [],
        roles: Array.isArray(data.filters?.roles) ? data.filters.roles : [],
      });
      setSelectedAuditId((current) => {
        if (current && list.some((item) => item.id === current)) {
          return current;
        }
        return list.length ? list[0].id : null;
      });
    } catch (loadError) {
      setAudits([]);
      if (loadError.code !== "NO_SESSION") {
        setError("Unable to load audit events");
      }
    } finally {
      setLoading(false);
    }
  }, [apiFetch, filters.action, filters.escalation, filters.namespace, filters.role, page, pageSize, resetDetail, sessionActive]);

  useEffect(() => {
    loadAudits();
  }, [loadAudits]);

  useEffect(() => {
    setPage(1);
  }, [filters.action, filters.escalation, filters.namespace, filters.role]);

  const fetchDetail = useCallback(
    async (auditId) => {
      if (!sessionActive || auditId === null || auditId === undefined) {
        resetDetail();
        return;
      }
      setDetailLoading(true);
      setDetailError("");
      try {
        const response = await apiFetch(`/governance/audits/${auditId}`);
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          setDetail(null);
          setDetailError(data.message || "Unable to load audit detail");
          return;
        }
        setDetail(data.audit || null);
      } catch (detailError) {
        if (detailError.code === "NO_SESSION") {
          setDetailError("Session expired. Sign in again to inspect audit history.");
        } else {
          setDetailError("Unable to load audit detail");
        }
        setDetail(null);
      } finally {
        setDetailLoading(false);
      }
    },
    [apiFetch, resetDetail, sessionActive],
  );

  useEffect(() => {
    if (selectedAuditId !== null && selectedAuditId !== undefined) {
      fetchDetail(selectedAuditId);
    } else {
      resetDetail();
    }
  }, [fetchDetail, resetDetail, selectedAuditId]);

  const totalPages = Math.max(1, Math.ceil((total || 0) / pageSize));
  const hasPrevious = page > 1;
  const hasNext = page * pageSize < total;

  const goPrevious = useCallback(() => {
    if (hasPrevious) {
      setPage((value) => Math.max(1, value - 1));
    }
  }, [hasPrevious]);

  const goNext = useCallback(() => {
    if (hasNext) {
      setPage((value) => value + 1);
    }
  }, [hasNext]);

  const handleExport = useCallback(() => {
    if (!detail) {
      return;
    }
    const blob = new Blob([JSON.stringify(detail, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `policy-audit-${detail.id || "event"}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    if (notify) {
      notify({ type: "success", message: "Audit exported" });
    }
  }, [detail, notify]);

  const handleRaiseTicket = useCallback(() => {
    if (!detail) {
      return;
    }
    const subject = encodeURIComponent(
      `Follow-up: ${detail.policy_name || detail.action || "policy event"}`,
    );
    const summary = detail.details
      ? JSON.stringify(detail.details, null, 2)
      : "See audit event for details.";
    const body = encodeURIComponent(
      `Policy: ${detail.policy_name}\nNamespace: ${detail.namespace}\nAction: ${detail.action}\nEscalation: ${detail.escalate_to || "none"}\n\nDetails:\n${summary}`,
    );
    try {
      window.open(`mailto:?subject=${subject}&body=${body}`, "_blank");
    } catch (err) {
      if (notify) {
        notify({ type: "error", message: "Unable to open mail client" });
      }
    }
  }, [detail, notify]);

  if (!sessionActive) {
    return (
      <section className="section">
        <h2>Policy audit explorer</h2>
        <div className="card placeholder-card">
          <p>Sign in to browse enforcement events and escalation history.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="section">
      <h2>Policy audit explorer</h2>
      <div className="card audit-card">
        <div className="card-header">
          <div>
            <h3>Enforcement events</h3>
            <p className="card-subtitle">
              Filter policy actions, preview payload diffs, and export audit artifacts.
            </p>
          </div>
        </div>
        <div className="audit-controls">
          <label>
            Action
            <select
              value={filters.action}
              onChange={(event) =>
                setFilters((prev) => ({ ...prev, action: event.target.value }))
              }
            >
              <option value="all">All actions</option>
              {filterOptions.actions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Namespace glob
            <input
              type="search"
              value={filters.namespace}
              onChange={(event) =>
                setFilters((prev) => ({ ...prev, namespace: event.target.value }))
              }
              placeholder="ops/*"
            />
          </label>
          <label>
            Escalation queue
            <select
              value={filters.escalation}
              onChange={(event) =>
                setFilters((prev) => ({ ...prev, escalation: event.target.value }))
              }
            >
              <option value="all">All queues</option>
              {filterOptions.escalations.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Role / team
            <select
              value={filters.role}
              onChange={(event) =>
                setFilters((prev) => ({ ...prev, role: event.target.value }))
              }
            >
              <option value="all">All</option>
              {filterOptions.roles.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Page size
            <select
              value={pageSize}
              onChange={(event) => setPageSize(Number(event.target.value) || 25)}
            >
              {[10, 25, 50, 100].map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
          <div className="audit-pagination">
            <button
              type="button"
              className="secondary"
              onClick={goPrevious}
              disabled={!hasPrevious || loading}
            >
              Previous
            </button>
            <span>
              Page {page} / {totalPages}
            </span>
            <button
              type="button"
              className="secondary"
              onClick={goNext}
              disabled={!hasNext || loading}
            >
              Next
            </button>
          </div>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="audit-layout">
          <div className="audit-table-wrapper">
            <div className="table-wrapper">
              <table className="data-table audit-table">
                <thead>
                  <tr>
                    <th scope="col">Timestamp</th>
                    <th scope="col">Namespace</th>
                    <th scope="col">Policy</th>
                    <th scope="col">Action</th>
                    <th scope="col">Escalation</th>
                  </tr>
                </thead>
                <tbody>
                  {audits.length ? (
                    audits.map((audit) => (
                      <tr
                        key={audit.id}
                        className={audit.id === selectedAuditId ? "selected" : ""}
                        onClick={() => setSelectedAuditId(audit.id)}
                      >
                        <td>{formatAuditTimestamp(audit.timestamp)}</td>
                        <td>{audit.namespace}</td>
                        <td>{audit.policy_name}</td>
                        <td>{audit.action}</td>
                        <td>{audit.escalate_to || "—"}</td>
                      </tr>
                    ))
                  ) : (
                    <tr className="empty-row">
                      <td colSpan={5}>
                        {loading ? "Loading audit events…" : "No audit events found"}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
          <div className="audit-detail">
            {detailLoading ? <p className="muted">Loading detail…</p> : null}
            {detailError ? <p className="error-text">{detailError}</p> : null}
            {detail ? (
              <div className="detail-card">
                <h3>{detail.policy_name}</h3>
                <div className="detail-meta">
                  <span>{formatAuditTimestamp(detail.timestamp)}</span>
                  <span>{detail.namespace}</span>
                  <span>Action: {detail.action}</span>
                </div>
                <div className="detail-section">
                  <h4>Violations</h4>
                  {Array.isArray(detail.violations) && detail.violations.length ? (
                    <ul>
                      {detail.violations.map((violation, index) => (
                        <li key={`${detail.id}-violation-${index}`}>{violation}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">No explicit violations recorded.</p>
                  )}
                </div>
                <div className="detail-section">
                  <h4>Escalation trail</h4>
                  <p className="muted">{detail.escalate_to || "No escalation queue triggered."}</p>
                </div>
                <div className="detail-section">
                  <h4>Payload diff</h4>
                  <pre className="audit-json-block">
                    {detail.details ? JSON.stringify(detail.details, null, 2) : "{}"}
                  </pre>
                </div>
                <div className="detail-actions">
                  <button type="button" className="secondary" onClick={handleExport}>
                    Export JSON
                  </button>
                  <button type="button" className="primary" onClick={handleRaiseTicket}>
                    Raise follow-up ticket
                  </button>
                </div>
              </div>
            ) : !detailLoading && !detailError ? (
              <p className="muted">Select an audit event to inspect payload details.</p>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}

function AnalyticsPanel({ apiFetch, sessionActive }) {
  const [days, setDays] = useState(30);
  const [top, setTop] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [refreshIndex, setRefreshIndex] = useState(0);

  const integerFormatter = useMemo(() => new Intl.NumberFormat(), []);
  const averageFormatter = useMemo(
    () => new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }),
    [],
  );

  const loadAnalytics = useCallback(async () => {
    if (!sessionActive) {
      setData(null);
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        days: String(days),
        top: String(top),
      });
      const response = await apiFetch(`/analytics/summary?${params.toString()}`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(payload.message || "Unable to load analytics");
        setData(null);
        return;
      }
      setData(payload);
    } catch (err) {
      setError(err.message || "Network error");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [apiFetch, sessionActive, days, top, refreshIndex]);

  useEffect(() => {
    loadAnalytics();
  }, [loadAnalytics]);

  const refresh = useCallback(() => {
    setRefreshIndex((value) => value + 1);
  }, []);

  const categorySections = useMemo(() => {
    if (!data?.category_counts) {
      return [];
    }
    return Object.entries(data.category_counts).map(([memoryType, categories]) => ({
      memoryType,
      total: categories.reduce((sum, entry) => sum + entry.count, 0),
      categories,
    }));
  }, [data]);

  const retentionSections = useMemo(() => {
    if (!data?.retention_trends) {
      return [];
    }
    const end = data.retention_trends?.range?.end;
    return [
      { key: "long_term", label: "Long-term" },
      { key: "short_term", label: "Short-term" },
    ]
      .map(({ key, label }) => {
        const series = data.retention_trends[key];
        if (!Array.isArray(series) || !series.length) {
          return null;
        }
        const entries = series.map((item) => {
          const values = Object.values(item.daily_counts || {});
          const total = values.reduce((sum, value) => sum + value, 0);
          const latest =
            end && item.daily_counts ? item.daily_counts[end] || 0 : 0;
          return {
            retentionType: item.retention_type,
            total,
            latest,
          };
        });
        return { key, label, entries };
      })
      .filter(Boolean);
  }, [data]);

  const usageSections = useMemo(() => {
    if (!data?.usage_frequency) {
      return [];
    }
    return Object.entries(data.usage_frequency).map(([key, section]) => ({
      key,
      totalRecords: section.total_records,
      totalAccesses: section.total_accesses,
      averageAccesses: section.average_accesses,
      topRecords: section.top_records || [],
    }));
  }, [data]);

  const hasAnalytics = Boolean(
    categorySections.length || retentionSections.length || usageSections.length,
  );

  return (
    <section className="section">
      <h2>Analytics</h2>
      <div className="card-grid">
        <div className="card">
          <div className="card-header">
            <h3>Controls</h3>
          </div>
          <div className="card-content">
            <div className="form-grid">
              <label>
                <span>Trend window</span>
                <select
                  value={days}
                  onChange={(event) => setDays(Number(event.target.value) || 30)}
                >
                  <option value={7}>7 days</option>
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                </select>
              </label>
              <label>
                <span>Top memories</span>
                <select
                  value={top}
                  onChange={(event) => setTop(Number(event.target.value) || 5)}
                >
                  <option value={5}>Top 5</option>
                  <option value={10}>Top 10</option>
                  <option value={20}>Top 20</option>
                </select>
              </label>
            </div>
            <button className="secondary" onClick={refresh} disabled={loading}>
              Refresh
            </button>
            {error ? <p className="error-text">{error}</p> : null}
            {loading ? <p className="muted">Loading analytics…</p> : null}
          </div>
        </div>
      </div>
      {!sessionActive ? (
        <div className="card placeholder-card">
          <p>Sign in to view analytics.</p>
        </div>
      ) : null}
      {sessionActive && !loading && !error && !hasAnalytics ? (
        <div className="card placeholder-card">
          <p>No analytics available yet. Add a few memories to populate stats.</p>
        </div>
      ) : null}
      {sessionActive && hasAnalytics ? (
        <div className="card-grid">
          {categorySections.map((section) => (
            <div className="card" key={`category-${section.memoryType}`}>
              <div className="card-header">
                <h3>
                  {section.memoryType === "long_term"
                    ? "Long-term categories"
                    : "Short-term categories"}
                </h3>
                <span className="muted">
                  Total {integerFormatter.format(section.total)} memories
                </span>
              </div>
              <div className="card-content">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Category</th>
                      <th scope="col">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.categories.map((category) => (
                      <tr key={`${section.memoryType}-${category.category}`}>
                        <th scope="row">{category.category}</th>
                        <td>{integerFormatter.format(category.count)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
          {retentionSections.map((section) => (
            <div className="card" key={`retention-${section.key}`}>
              <div className="card-header">
                <h3>{section.label} retention</h3>
                {data?.retention_trends?.range ? (
                  <span className="muted">
                    {data.retention_trends.range.start} → {" "}
                    {data.retention_trends.range.end}
                  </span>
                ) : null}
              </div>
              <div className="card-content">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Retention type</th>
                      <th scope="col">Total</th>
                      <th scope="col">Most recent day</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.entries.map((entry) => (
                      <tr key={`${section.key}-${entry.retentionType}`}>
                        <th scope="row">{entry.retentionType}</th>
                        <td>{integerFormatter.format(entry.total)}</td>
                        <td>{integerFormatter.format(entry.latest)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
          {usageSections.map((section) => (
            <div className="card" key={`usage-${section.key}`}>
              <div className="card-header">
                <h3>
                  {section.key === "long_term"
                    ? "Long-term usage"
                    : "Short-term usage"}
                </h3>
                <span className="muted">
                  {integerFormatter.format(section.totalRecords)} records · {" "}
                  {integerFormatter.format(section.totalAccesses)} total touches · {" "}
                  {averageFormatter.format(section.averageAccesses)} avg
                </span>
              </div>
              <div className="card-content">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Memory</th>
                      <th scope="col">Category</th>
                      <th scope="col">Accesses</th>
                      <th scope="col">Last accessed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.topRecords.map((record) => (
                      <tr key={record.memory_id}>
                        <th scope="row">
                          {record.summary || record.memory_id || "(untitled)"}
                        </th>
                        <td>{record.category || "–"}</td>
                        <td>{integerFormatter.format(record.access_count)}</td>
                        <td>
                          {record.last_accessed
                            ? new Date(record.last_accessed).toLocaleString()
                            : "–"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function ClusterTools({ apiFetch, sessionActive }) {
  return (
    <section className="section">
      <h2>Cluster Operations</h2>
      <div className="card-grid">
        <Rebuild apiFetch={apiFetch} sessionActive={sessionActive} />
        <Activity apiFetch={apiFetch} sessionActive={sessionActive} />
      </div>
      <ClusterList apiFetch={apiFetch} sessionActive={sessionActive} />
    </section>
  );
}

function App() {
  const [storedApiKey, setStoredApiKey] = useStoredApiKey();
  const hasStoredKey = Boolean(storedApiKey);
  const [sessionActive, setSessionActive] = useState(() => hasStoredKey);
  const [sessionRequiresKey, setSessionRequiresKey] = useState(
    () => hasStoredKey,
  );
  const [loginVisible, setLoginVisible] = useState(() => !hasStoredKey);
  const [loginError, setLoginError] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [toasts, setToasts] = useState([]);
  const toastTimersRef = useRef(new Map());

  const dismissToast = useCallback((id) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
    const timers = toastTimersRef.current;
    if (timers.has(id)) {
      clearTimeout(timers.get(id));
      timers.delete(id);
    }
  }, []);

  const pushToast = useCallback((toast) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const next = {
      id,
      type: toast.type === "error" ? "error" : "success",
      message: toast.message,
    };
    setToasts((current) => [...current, next]);
    const duration = toast.duration || (next.type === "error" ? 6000 : 4000);
    const timeoutId = setTimeout(() => dismissToast(id), duration);
    toastTimersRef.current.set(id, timeoutId);
  }, [dismissToast]);

  useEffect(() => {
    return () => {
      toastTimersRef.current.forEach((timer) => clearTimeout(timer));
      toastTimersRef.current.clear();
    };
  }, []);

  useEffect(() => {
    if (storedApiKey) {
      setSessionActive(true);
    }
  }, [storedApiKey]);

  useEffect(() => {
    if (!storedApiKey && sessionRequiresKey) {
      setSessionActive(false);
    }
  }, [storedApiKey, sessionRequiresKey]);

  useEffect(() => {
    setLoginVisible(!sessionActive);
  }, [sessionActive]);

  const handleUnauthorized = useCallback(
    (message) => {
      setStoredApiKey("");
      setSessionActive(false);
      setSessionRequiresKey(false);
      setLoginError(message || "Session expired. Please sign in again.");
    },
    [setStoredApiKey, setSessionActive, setSessionRequiresKey],
  );

  const apiFetch = useCallback(
    async (input, init = {}) => {
      if (!sessionActive) {
        const error = new Error("Missing session");
        error.code = "NO_SESSION";
        throw error;
      }
      if (sessionRequiresKey && !storedApiKey) {
        const error = new Error("Missing API key");
        error.code = "NO_SESSION";
        throw error;
      }
      const headers = new Headers(init.headers || {});
      if (sessionRequiresKey && storedApiKey) {
        headers.set("X-API-Key", storedApiKey);
      } else if (!sessionRequiresKey) {
        headers.delete("X-API-Key");
      }
      const response = await fetch(input, { ...init, headers });
      if (response.status === 401) {
        const data = await response
          .clone()
          .json()
          .catch(() => ({}));
        handleUnauthorized(data.message);
        throw new Error("Unauthorized");
      }
      return response;
    },
    [sessionActive, sessionRequiresKey, storedApiKey, handleUnauthorized],
  );

  const submitLogin = useCallback(
    async (value) => {
      const apiKey = value || "";
      setLoginBusy(true);
      setLoginError("");
      try {
        const response = await fetch("/session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: apiKey }),
        });
        const data = await response.json().catch(() => ({}));
        if (response.status === 401) {
          setLoginError(data.message || "Invalid API key.");
          return;
        }
        if (!response.ok) {
          setLoginError(data.message || "Unable to start session.");
          return;
        }
        const requiresKey = Object.prototype.hasOwnProperty.call(
          data,
          "requires_key",
        )
          ? Boolean(data.requires_key)
          : Boolean(apiKey);
        if (apiKey) {
          setStoredApiKey(apiKey);
        }
        setSessionRequiresKey(requiresKey);
        setSessionActive(true);
        setLoginVisible(false);
        setLoginError("");
      } catch (error) {
        setLoginError("Network error. Please try again.");
      } finally {
        setLoginBusy(false);
      }
    },
    [
      setStoredApiKey,
      setSessionActive,
      setSessionRequiresKey,
      setLoginVisible,
    ],
  );

  const logout = useCallback(() => {
    setStoredApiKey("");
    setLoginError("");
    setSessionActive(false);
    setSessionRequiresKey(false);
  }, [setStoredApiKey, setSessionActive, setSessionRequiresKey]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Memoria Dashboard</h1>
          <p className="subtitle">
            Inspect stored memories, governance policies, and cluster activity.
          </p>
        </div>
        <div className="session-actions">
          {sessionActive ? (
            <button className="secondary" onClick={logout}>
              Log out
            </button>
          ) : (
            <button className="primary" onClick={() => setLoginVisible(true)}>
              Sign in
            </button>
          )}
        </div>
      </header>
      <SettingsPanel
        apiFetch={apiFetch}
        sessionActive={sessionActive}
        notify={pushToast}
      />
      <NamespaceSegmentationPanel
        apiFetch={apiFetch}
        sessionActive={sessionActive}
      />
      <EscalationRosterPanel
        apiFetch={apiFetch}
        sessionActive={sessionActive}
        notify={pushToast}
      />
      <EnforcementTelemetryPanel
        apiFetch={apiFetch}
        sessionActive={sessionActive}
      />
      <PolicyOverridePanel
        apiFetch={apiFetch}
        sessionActive={sessionActive}
        notify={pushToast}
      />
      <PolicyAuditExplorer
        apiFetch={apiFetch}
        sessionActive={sessionActive}
        notify={pushToast}
      />
      <TableBrowser apiFetch={apiFetch} sessionActive={sessionActive} />
      <AnalyticsPanel apiFetch={apiFetch} sessionActive={sessionActive} />
      <ClusterTools apiFetch={apiFetch} sessionActive={sessionActive} />
      <LoginModal
        visible={loginVisible}
        busy={loginBusy}
        errorMessage={loginError}
        onSubmit={submitLogin}
      />
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

const rootElement = document.getElementById("root");
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(<App />);
}
