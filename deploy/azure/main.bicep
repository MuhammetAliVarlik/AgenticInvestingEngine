// Investing Engine on Azure Container Apps (consumption plan).
//
// Cost model: both apps scale to zero and run at most one replica, which keeps
// usage inside the consumption plan's monthly free grant. Images are pulled
// from GitHub Container Registry (no Azure Container Registry fee) and no Log
// Analytics workspace is created (use `az containerapp logs show` instead).
//
// Security model: only the UI has public ingress, and it sits behind Container
// Apps built-in authentication (GitHub sign-in). The API has internal ingress
// only and additionally requires the shared internal token and an allowlisted
// user identity on every request.

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Name prefix for all resources.')
@minLength(3)
@maxLength(16)
param prefix string = 'invengine'

@description('Container images, e.g. ghcr.io/<owner>/investing-engine-api:<tag>.')
param apiImage string
param uiImage string

@description('Comma-separated allowlist, e.g. "github:octocat,alice@example.com".')
param allowedUsers string

@description('GitHub OAuth app client id used by Container Apps authentication.')
param githubClientId string

@secure()
param githubClientSecret string

@secure()
@minLength(32)
param internalApiToken string

@secure()
param groqApiKey string

@secure()
param evdsApiKey string

@secure()
param telemetrySalt string

param langfusePublicKey string = ''

@secure()
param langfuseSecretKey string = ''

@description('Enable the Llama Prompt Guard classifier on Groq.')
param enablePromptGuard bool = false

var apiName = '${prefix}-api'
var uiName = '${prefix}-ui'

var apiSecrets = concat(
  [
    { name: 'internal-api-token', value: internalApiToken }
    { name: 'groq-api-key', value: groqApiKey }
    { name: 'evds-api-key', value: evdsApiKey }
    { name: 'telemetry-salt', value: telemetrySalt }
  ],
  empty(langfuseSecretKey) ? [] : [{ name: 'langfuse-secret-key', value: langfuseSecretKey }]
)

var apiEnv = concat(
  [
    { name: 'ENVIRONMENT', value: 'production' }
    { name: 'AUTH_MODE', value: 'trusted-proxy' }
    { name: 'ALLOWED_USERS', value: allowedUsers }
    { name: 'INTERNAL_API_TOKEN', secretRef: 'internal-api-token' }
    { name: 'LLM_PROVIDER', value: 'groq' }
    { name: 'GROQ_API_KEY', secretRef: 'groq-api-key' }
    { name: 'EVDS_API_KEY', secretRef: 'evds-api-key' }
    { name: 'TELEMETRY_SALT', secretRef: 'telemetry-salt' }
    { name: 'ENABLE_PROMPT_GUARD', value: string(enablePromptGuard) }
    { name: 'ENABLE_YFINANCE', value: 'false' }
    { name: 'DB_PATH', value: '/tmp/data/predictions.db' }
    { name: 'MODEL_DIR', value: '/tmp/data/models' }
  ],
  empty(langfuseSecretKey)
    ? []
    : [
        { name: 'LANGFUSE_PUBLIC_KEY', value: langfusePublicKey }
        { name: 'LANGFUSE_SECRET_KEY', secretRef: 'langfuse-secret-key' }
      ]
)

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  properties: {
    workloadProfiles: [
      { name: 'Consumption', workloadProfileType: 'Consumption' }
    ]
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: apiName
  location: location
  properties: {
    environmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false // reachable only from inside the environment
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      secrets: apiSecrets
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('0.75'), memory: '1.5Gi' }
          env: apiEnv
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/healthz', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          { name: 'http', http: { metadata: { concurrentRequests: '20' } } }
        ]
      }
    }
  }
}

resource ui 'Microsoft.App/containerApps@2024-03-01' = {
  name: uiName
  location: location
  properties: {
    environmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto' // HTTP/1.1 + WebSockets for Streamlit
        allowInsecure: false
        stickySessions: { affinity: 'sticky' }
      }
      secrets: [
        { name: 'internal-api-token', value: internalApiToken }
        { name: 'github-client-secret', value: githubClientSecret }
      ]
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: uiImage
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          env: [
            { name: 'API_BASE_URL', value: 'http://${apiName}' }
            { name: 'AUTH_PROVIDER', value: 'easyauth' }
            { name: 'ALLOWED_USERS', value: allowedUsers }
            { name: 'INTERNAL_API_TOKEN', secretRef: 'internal-api-token' }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

// Built-in authentication: every request to the UI must be signed in with GitHub.
resource uiAuth 'Microsoft.App/containerApps/authConfigs@2024-03-01' = {
  parent: ui
  name: 'current'
  properties: {
    platform: { enabled: true }
    globalValidation: {
      unauthenticatedClientAction: 'RedirectToLoginPage'
      redirectToProvider: 'github'
    }
    identityProviders: {
      gitHub: {
        enabled: true
        registration: {
          clientId: githubClientId
          clientSecretSettingName: 'github-client-secret'
        }
      }
    }
    login: {
      preserveUrlFragmentsForLogins: false
    }
  }
}

output uiUrl string = 'https://${ui.properties.configuration.ingress.fqdn}'
output githubCallbackUrl string = 'https://${ui.properties.configuration.ingress.fqdn}/.auth/login/github/callback'
