# GCP TPU Zones Quota & Startup Status

This file details the GCP zones with available quota for `TPUV6EPerProjectPerZoneForTPUAPI` and the status of TPU startup attempts for `v6e-1`.

| Zone | Quota Available | TPU v6e-1 Started Successfully | Details / Reason for Failure |
| :--- | :--- | :--- | :--- |
| **asia-east1-a** | Yes | No | ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested accelerator configuration for accelerator type, "v6e-1", could not be found in the zonal accelerator configurations for "asia-east1-a".; the accelerator v6e-1 was not found in zone asia-east1-a [EID: 0x3e57d081b512151f]. Cleaned up: [] |
| **asia-east1-b** | Yes | No | ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested accelerator configuration for accelerator type, "v6e-1", could not be found in the zonal accelerator configurations for "asia-east1-b".; the accelerator v6e-1 was not found in zone asia-east1-b [EID: 0xaccdf84e55ae951a]. Cleaned up: [] |
| **asia-east1-c** | Yes | No | Create request issued for: [vllm-gemma4-qr]
Waiting for operation [projects/comglitn/locations/asia-east1-c/operations/operation-1783349881865-655f280380433-905db7ef-6ad4e8f0] to complete...
...........failed.
ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) {
  "code": 3,
  "message": "FLEX_START provisioning model is not supported for accelerator type \"v6e-1\" in location \"asia-east1-c\""
}. Cleaned up: [] |
| **asia-northeast1-a** | Yes | No | ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested zone "asia-northeast1-a" is not supported. [EID: 0xd782146204d09293]. Cleaned up: [] |
| **asia-northeast1-b** | Yes | No | Timed out waiting 3 minutes to reach ACTIVE state. |
| **asia-northeast1-c** | Yes | No | ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested zone "asia-northeast1-c" is not supported. [EID: 0xef5b50c9b34850dc]. Cleaned up: [] |
| **asia-south1-a** | Yes | No | ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested zone "asia-south1-a" is not supported. [EID: 0xa9fd7efe8466e92c]. Cleaned up: [] |
| **asia-south1-b** | Yes | No | Create request issued for: [vllm-gemma4-qr]
Waiting for operation [projects/comglitn/locations/asia-south1-b/operations/operation-1783350119803-655f28e66a945-16331f04-7ebb11ad] to complete...
..............failed.
ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) {
  "code": 3,
  "message": "FLEX_START provisioning model is not supported for accelerator type \"v6e-1\" in location \"asia-south1-b\""
}. Cleaned up: [] |
| **asia-south1-c** | Yes | No | Create request issued for: [vllm-gemma4-qr]
Waiting for operation [projects/comglitn/locations/asia-south1-c/operations/operation-1783350133990-655f28f3f24f2-73b1ccf9-e49f0926] to complete...
.............failed.
ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) {
  "code": 3,
  "message": "FLEX_START provisioning model is not supported for accelerator type \"v6e-1\" in location \"asia-south1-c\""
}. Cleaned up: [] |
| **asia-southeast1-a** | Yes | No | ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested zone "asia-southeast1-a" is not supported. [EID: 0x8ad2a7734dc5072d]. Cleaned up: [] |
| **asia-southeast1-b** | Yes | No | Create request issued for: [vllm-gemma4-qr]
Waiting for operation [projects/comglitn/locations/asia-southeast1-b/operations/operation-1783350154926-655f2907e987b-654844f5-bb1bede0] to complete...
............failed.
ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) {
  "code": 3,
  "message": "FLEX_START provisioning model is not supported for accelerator type \"v6e-1\" in location \"asia-southeast1-b\""
}. Cleaned up: [] |
| **asia-southeast1-c** | Yes | No | ERROR: (gcloud.alpha.compute.tpus.queued-resources.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested zone "asia-southeast1-c" is not supported. [EID: 0x72f053208118f800]. Cleaned up: [] |
| **europe-west4-a** | Yes | Yes | Successfully started and reached ACTIVE state. |
