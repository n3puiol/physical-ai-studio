import { SchemaDeviceInfo } from '../../../api/openapi-spec';
import { resolveGpuSelection } from './gpu-selection';

const cuda = (index: number): SchemaDeviceInfo => ({ type: 'cuda', index, name: `GPU ${index}` });
const threeGpus = [cuda(0), cuda(1), cuda(2)];
const both = ['all_devices', 'device_indices'];

describe('resolveGpuSelection', () => {
    it.each([
        { keys: ['cuda:1'], capabilities: [], kind: 'single', indices: [1] },
        { keys: ['cuda:0', 'cuda:1', 'cuda:2'], capabilities: ['all_devices'], kind: 'all', indices: [0, 1, 2] },
        // Every GPU prefers all_devices, which older trainer images understand too.
        { keys: ['cuda:2', 'cuda:0', 'cuda:1'], capabilities: both, kind: 'all', indices: [0, 1, 2] },
        { keys: ['cuda:2', 'cuda:0'], capabilities: both, kind: 'indices', indices: [0, 2] },
        { keys: ['cuda:0', 'cuda:1', 'cuda:2'], capabilities: ['device_indices'], kind: 'indices', indices: [0, 1, 2] },
    ])('resolves $keys with $capabilities to $kind', ({ keys, capabilities, kind, indices }) => {
        const selection = resolveGpuSelection(keys, threeGpus, capabilities);
        expect(selection.kind).toBe(kind);
        expect(selection.devices.map((device) => device.index)).toEqual(indices);
    });

    it.each([
        { keys: [], capabilities: both, message: 'Choose at least one GPU.' },
        { keys: ['cuda:0', 'cuda:7'], capabilities: both, message: /no longer available/ },
        { keys: ['cuda:0', 'cuda:1'], capabilities: [], message: /only train on one GPU at a time/ },
        { keys: ['cuda:0', 'cuda:1'], capabilities: ['all_devices'], message: /one GPU or on all of them/ },
    ])('rejects $keys with $capabilities', ({ keys, capabilities, message }) => {
        const selection = resolveGpuSelection(keys, threeGpus, capabilities);
        expect(selection.kind).toBe('invalid');
        expect(selection.kind === 'invalid' && selection.message).toMatch(message);
    });

    it('rejects several GPUs that are not all CUDA GPUs', () => {
        const devices: SchemaDeviceInfo[] = [cuda(0), { type: 'xpu', index: 0, name: 'Arc' }];
        const selection = resolveGpuSelection(['cuda:0', 'xpu:0'], devices, both);
        expect(selection.kind === 'invalid' && selection.message).toBe('Only CUDA GPUs can train one job together.');
    });
});
