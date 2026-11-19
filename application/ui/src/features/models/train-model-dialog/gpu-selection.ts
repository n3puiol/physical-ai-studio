import { SchemaDeviceInfo } from '../../../api/openapi-spec';

/** How a set of ticked GPUs is sent to a remote trainer, or why it can't be. */
export type GpuSelection =
    | { kind: 'single'; devices: [SchemaDeviceInfo] }
    /** Every CUDA GPU of the trainer: sent as `all_devices`, which older trainer images understand too. */
    | { kind: 'all'; devices: SchemaDeviceInfo[] }
    /** Some CUDA GPUs of the trainer: sent as `indices`. */
    | { kind: 'indices'; devices: SchemaDeviceInfo[] }
    | { kind: 'invalid'; devices: SchemaDeviceInfo[]; message: string };

export const gpuKey = (device: SchemaDeviceInfo) => `${device.type}:${device.index}`;

/** Whether the trainer can train one job on several of its GPUs at all. */
export const supportsSeveralGpus = (capabilities: readonly string[]) =>
    capabilities.includes('all_devices') || capabilities.includes('device_indices');

/**
 * Turn the ticked GPU keys into what the job trains on.
 *
 * One GPU trains as before. Several must all be CUDA GPUs: every one of them is
 * `all_devices`, otherwise the trainer must support `device_indices`.
 */
export const resolveGpuSelection = (
    keys: readonly string[],
    devices: readonly SchemaDeviceInfo[],
    capabilities: readonly string[]
): GpuSelection => {
    const selected = devices.filter((device) => keys.includes(gpuKey(device)));
    if (selected.length !== new Set(keys).size) {
        return {
            kind: 'invalid',
            devices: selected,
            message: 'A selected GPU is no longer available. Choose another GPU.',
        };
    }
    if (selected.length === 0) {
        return { kind: 'invalid', devices: selected, message: 'Choose at least one GPU.' };
    }
    if (selected.length === 1) {
        return { kind: 'single', devices: [selected[0]] };
    }
    if (selected.some((device) => device.type !== 'cuda')) {
        return { kind: 'invalid', devices: selected, message: 'Only CUDA GPUs can train one job together.' };
    }
    const cudaCount = devices.filter((device) => device.type === 'cuda').length;
    if (selected.length === cudaCount && capabilities.includes('all_devices')) {
        return { kind: 'all', devices: selected };
    }
    if (capabilities.includes('device_indices')) {
        return { kind: 'indices', devices: selected };
    }
    return {
        kind: 'invalid',
        devices: selected,
        message: capabilities.includes('all_devices')
            ? "This trainer's image can train on one GPU or on all of them. Update it to choose some."
            : "This trainer's image can only train on one GPU at a time. Update it to train on several.",
    };
};
