{
  "$schema": "https://mintlify.com/docs.json",
  "theme": "mint",
  "name": "Beam",
  "colors": {
    "primary": "#477ff7",
    "light": "#93c5fd",
    "dark": "#2563eb"
  },
  "favicon": "/logo/favicon.png",
  "navigation": {
    "tabs": [
      {
        "tab": "Documentation",
        "groups": [
          {
            "group": "Start Here",
            "pages": [
              "v2/getting-started/introduction",
              "v2/getting-started/installation",
              "v2/getting-started/quickstart",
              "v2/getting-started/core-concepts"
            ]
          },
          {
            "group": "Customizing Container Images",
            "pages": [
              "v2/environment/custom-images",
              "v2/environment/custom-registries",
              "v2/environment/gpu",
              "v2/environment/resources",
              "v2/environment/secrets"
            ]
          },
          {
            "group": "Running Arbitrary Code",
            "pages": [
              "v2/pod/overview",
              "v2/pod/streamlit",
              "v2/pod/web_server"
            ]
          },
          {
            "group": "Managing Data",
            "pages": [
              "v2/data/volume",
              "v2/data/external-storage",
              "v2/data/output"
            ]
          },
          {
            "group": "Autoscaling and Concurrency",
            "pages": [
              "v2/scaling/concurrency",
              "v2/scaling/concurrent-inputs",
              "v2/scaling/parallelizing-functions"
            ]
          },
          {
            "group": "Endpoints and Web Servers",
            "pages": [
              "v2/endpoint/overview",
              "v2/endpoint/web-server",
              "v2/endpoint/realtime",
              "v2/endpoint/loaders",
              "v2/endpoint/keep-warm",
              "v2/endpoint/versioning",
              "v2/endpoint/sending-file-payloads"
            ]
          },
          {
            "group": "Task Queues",
            "pages": [
              "v2/task-queue/running-tasks",
              "v2/task-queue/query-status"
            ]
          },
          {
            "group": "Serverless Functions",
            "pages": [
              "v2/function/running-functions",
              "v2/function/scheduled-job",
              "v2/function/queues",
              "v2/function/maps"
            ]
          },
          {
            "group": "Agents (New)",
            "pages": ["v2/agents/introduction", "v2/agents/synchronization"]
          },
          {
            "group": "Other Topics",
            "pages": [
              "v2/topics/ci",
              "v2/topics/timeouts-and-retries",
              "v2/topics/public-endpoints",
              "v2/topics/callbacks",
              "v2/topics/signal",
              "v2/topics/context",
              "v2/topics/cold-start",
              "v2/environment/jupyter-notebook",
              "v2/environment/remote-versus-local"
            ]
          },
          {
            "group": "Self-Hosting",
            "pages": [
              "v2/self-hosting/overview",
              "v2/self-hosting/local-machine",
              "v2/self-hosting/aws"
            ]
          },
          {
            "group": "Resources",
            "pages": [
              "v2/resources/pricing-and-billing",
              "v2/resources/faq",
              "v2/resources/cog"
            ]
          },
          {
            "group": "Security",
            "pages": [
              "v2/security/terms-and-conditions",
              "v2/security/privacy-policy"
            ]
          }
        ]
      },
      {
        "tab": "Examples",
        "groups": [
          {
            "group": "Infrastructure",
            "pages": ["v2/examples/web-scraping"]
          },
          {
            "group": "Large language models (LLMs)",
            "pages": [
              "v2/examples/deepseek-r1",
              "v2/examples/vllm",
              "v2/examples/inference",
              "v2/examples/llama3"
            ]
          },
          {
            "group": "Fine Tuning",
            "pages": ["v2/examples/gemma-fine-tune", "v2/examples/unsloth"]
          },
          {
            "group": "Images, Video",
            "pages": [
              "v2/examples/comfy-ui",
              "v2/examples/lora",
              "v2/examples/mochi-1"
            ]
          },
          {
            "group": "Audio",
            "pages": ["v2/examples/parler-tts", "v2/examples/whisper", "v2/examples/zonos"]
          }
        ]
      },
      {
        "tab": "Reference",
        "groups": [
          {
            "group": "Reference",
            "pages": [
              "v2/reference/sdk",
              "v2/reference/cli",
              "v2/reference/api",
              "v2/reference/node"
            ]
          }
        ]
      },
      {
        "tab": "Changelog",
        "groups": [
          {
            "group": "Changelog",
            "pages": [
              "v2/releases/02-26-25",
              "v2/releases/02-25-25",
              "v2/releases/02-24-25",
              "v2/releases/02-21-25",
              "v2/releases/02-19-25",
              "v2/releases/02-10-25",
              "v2/releases/02-08-25",
              "v2/releases/02-07-25",
              "v2/releases/02-06-25",
              "v2/releases/02-05-25",
              "v2/releases/02-04-25",
              "v2/releases/02-03-25",
              "v2/releases/02-02-25",
              "v2/releases/02-01-25",
              "v2/releases/01-30-25",
              "v2/releases/01-29-25",
              "v2/releases/01-28-25",
              "v2/releases/01-27-25",
              "v2/releases/01-26-25",
              "v2/releases/01-25-25",
              "v2/releases/01-24-25",
              "v2/releases/01-21-25",
              "v2/releases/01-20-25",
              "v2/releases/01-17-25",
              "v2/releases/01-16-25",
              "v2/releases/01-15-25",
              "v2/releases/01-14-25",
              "v2/releases/01-13-25",
              "v2/releases/01-11-25",
              "v2/releases/01-10-25",
              "v2/releases/01-09-25",
              "v2/releases/01-08-25",
              "v2/releases/01-07-25",
              "v2/releases/01-03-25",
              "v2/releases/01-02-25",
              "v2/releases/12-27-24",
              "v2/releases/12-20-24",
              "v2/releases/12-19-24",
              "v2/releases/12-18-24",
              "v2/releases/12-16-24",
              "v2/releases/12-12-24",
              "v2/releases/12-11-24",
              "v2/releases/12-10-24",
              "v2/releases/12-06-24",
              "v2/releases/12-04-24",
              "v2/releases/12-03-24",
              "v2/releases/11-30-24",
              "v2/releases/11-27-24",
              "v2/releases/11-25-24",
              "v2/releases/11-23-24",
              "v2/releases/11-22-24",
              "v2/releases/11-21-24",
              "v2/releases/11-19-24",
              "v2/releases/11-18-24",
              "v2/releases/11-14-24",
              "v2/releases/11-13-24",
              "v2/releases/11-12-24",
              "v2/releases/11-11-24",
              "v2/releases/11-07-24",
              "v2/releases/11-04-24",
              "v2/releases/11-03-24",
              "v2/releases/11-01-24",
              "v2/releases/10-31-24",
              "v2/releases/10-30-24",
              "v2/releases/10-29-24",
              "v2/releases/10-28-24",
              "v2/releases/10-24-24",
              "v2/releases/10-22-24",
              "v2/releases/10-21-24",
              "v2/releases/10-18-24",
              "v2/releases/10-17-24",
              "v2/releases/10-16-24",
              "v2/releases/10-15-24",
              "v2/releases/10-12-24",
              "v2/releases/10-11-24",
              "v2/releases/10-09-24",
              "v2/releases/10-08-24",
              "v2/releases/10-07-24",
              "v2/releases/09-23-24",
              "v2/releases/09-04-24",
              "v2/releases/08-08-24",
              "v2/releases/07-22-24",
              "v2/releases/07-11-24",
              "v2/releases/07-02-24",
              "v2/releases/06-24-24",
              "v2/releases/06-14-24",
              "v2/releases/v2-upgrade"
            ]
          }
        ]
      }
    ],
    "global": {
      "anchors": [
        {
          "anchor": "Slack Community",
          "href": "https://join.slack.com/t/beam-cloud/shared_invite/zt-2uiks0hc6-UbBD97oZjz8_YnjQ2P7BEQ",
          "icon": "slack",
          "color": {
            "light": "#6365f1",
            "dark": "#6365f1"
          }
        },
        {
          "anchor": "Github",
          "href": "https://github.com/beam-cloud/beta9",
          "icon": "github",
          "color": {
            "light": "#6365f1",
            "dark": "#6365f1"
          }
        },
        {
          "anchor": "Twitter",
          "href": "https://twitter.com/beam_cloud",
          "icon": "twitter",
          "color": {
            "light": "#6365f1",
            "dark": "#6365f1"
          }
        }
      ]
    }
  },
  "logo": {
    "light": "/logo/beam-logo-dark.svg",
    "dark": "/logo/beam-logo-light.svg"
  },
  "navbar": {
    "links": [
      {
        "label": "Dashboard",
        "href": "https://platform.beam.cloud"
      }
    ],
    "primary": {
      "type": "github",
      "href": "https://github.com/beam-cloud/beta9"
    }
  },
  "seo": {
    "metatags": {
      "og:image": "https://www.beam.cloud/meta-hero.png"
    },
    "indexing": "navigable"
  },
  "footer": {
    "socials": {
      "website": "https://beam.cloud",
      "github": "https://github.com/beam-cloud/beta9",
      "slack": "https://join.slack.com/t/beam-cloud/shared_invite/zt-2uiks0hc6-UbBD97oZjz8_YnjQ2P7BEQ"
    }
  },
  "integrations": {
    "ga4": {
      "measurementId": "G-XN1SS41L9X"
    },
    "logrocket": {
      "appId": "tmarkn/slai-production"
    }
  }
}
