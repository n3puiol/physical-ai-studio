import { Checkbox, CheckboxGroup, Flex, Item, Key, Picker, StatusLight, Text } from '@geti-ui/ui';

import { SchemaAvailablePolicy, SchemaDeviceInfo, SchemaProjectInput } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { gpuKey } from './gpu-selection';
import { formatBytes } from './policies';
import { PolicyAccessAlert } from './policy-access-alert';
import { PolicySelection } from './policy-selection';
import { TrainingTargetOption } from './train-model-dialog';

interface SetupStepProps {
    datasets: SchemaProjectInput['datasets'];
    selectedDataset: Key | null;
    onSelectedDatasetChange: (value: Key | null) => void;
    trainingTargetOptions: TrainingTargetOption[];
    targetId: Key | null;
    onTargetIdChange: (value: Key | null) => void;
    remoteDevices: SchemaDeviceInfo[];
    busyGpuKeys: Set<string>;
    /** Whether the GPU checklist is shown (the trainer has several GPUs, or one was ticked). */
    showGpuSelection: boolean;
    /** `type:index` keys of the ticked GPUs. */
    selectedGpuKeys: string[];
    onSelectedGpuKeysChange: (keys: string[]) => void;
    /** Whether "Auto-select all free GPUs" is offered (the trainer can train on a chosen set). */
    canAutoSelectGpus: boolean;
    autoSelectGpus: boolean;
    onAutoSelectGpusChange: (enabled: boolean) => void;
    /** Why the ticked GPUs can't train, if they can't. */
    gpuSelectionError: string | null;
    remoteUnavailable: boolean;
    selectedPolicy: string;
    availablePolicies: SchemaAvailablePolicy[];
    isLoadingPolicies: boolean;
    policiesUnavailable?: boolean;
    onSelectedPolicyChange: (policy: string) => void;
    isPolicyDisabled: boolean;
    activeDevice: SchemaDeviceInfo | null;
}

export const SetupStep = ({
    datasets,
    selectedDataset,
    onSelectedDatasetChange,
    trainingTargetOptions,
    targetId,
    onTargetIdChange,
    remoteDevices,
    busyGpuKeys,
    showGpuSelection,
    selectedGpuKeys,
    onSelectedGpuKeysChange,
    canAutoSelectGpus,
    autoSelectGpus,
    onAutoSelectGpusChange,
    gpuSelectionError,
    remoteUnavailable,
    selectedPolicy,
    availablePolicies,
    isLoadingPolicies,
    policiesUnavailable,
    onSelectedPolicyChange,
    isPolicyDisabled,
    activeDevice,
}: SetupStepProps) => (
    <Flex direction='column' gap='size-200' width='100%'>
        {remoteUnavailable && (
            <InlineAlert variant='warning'>
                Can&apos;t reach the remote trainer, so training can&apos;t start. Make sure it&apos;s running, then try
                again.
            </InlineAlert>
        )}

        <Picker label='Dataset' selectedKey={selectedDataset} onSelectionChange={onSelectedDatasetChange} width='100%'>
            {datasets.map((dataset) => (
                <Item key={dataset.id}>{dataset.name}</Item>
            ))}
        </Picker>

        <Picker
            label='Run on'
            selectedKey={targetId}
            onSelectionChange={onTargetIdChange}
            width='100%'
            items={trainingTargetOptions}
        >
            {(trainingTarget) => (
                <Item key={trainingTarget.id} textValue={trainingTarget.label}>
                    <Text>{trainingTarget.label}</Text>
                    {/* `Item` only recognizes plain `Text` children for its label/description
                        slots (a `StatusLight` isn't one), so nest it inside the description
                        slot rather than passing it as a sibling — otherwise both children
                        collapse into the same "label" grid area and overlap. */}
                    <Text slot='description'>{trainingTarget.statusLabel}</Text>
                    <Text slot='icon'>
                        <StatusLight variant={trainingTarget.statusVariant} marginBottom={0} />
                    </Text>
                </Item>
            )}
        </Picker>

        {showGpuSelection && (
            <Flex direction='column' gap='size-50' width='100%'>
                <CheckboxGroup
                    label='GPUs'
                    width='100%'
                    value={selectedGpuKeys}
                    onChange={onSelectedGpuKeysChange}
                    isDisabled={autoSelectGpus}
                    validationState={gpuSelectionError === null ? undefined : 'invalid'}
                    errorMessage={gpuSelectionError ?? undefined}
                    description={
                        selectedGpuKeys.length > 1 && gpuSelectionError === null
                            ? 'The job trains on the ticked GPUs together.'
                            : undefined
                    }
                >
                    {remoteDevices.map((device) => (
                        <Checkbox key={gpuKey(device)} value={gpuKey(device)}>
                            {`${device.type.toUpperCase()} ${device.index} — ${device.name}${
                                device.memory ? ` (${formatBytes(device.memory)})` : ''
                            }${busyGpuKeys.has(gpuKey(device)) ? ' · busy' : ''}`}
                        </Checkbox>
                    ))}
                </CheckboxGroup>
                {canAutoSelectGpus && (
                    <Checkbox isSelected={autoSelectGpus} onChange={onAutoSelectGpusChange}>
                        Auto-select all free GPUs
                    </Checkbox>
                )}
            </Flex>
        )}

        <PolicySelection
            selectedPolicy={selectedPolicy}
            availablePolicies={availablePolicies}
            isLoadingPolicies={isLoadingPolicies}
            policiesUnavailable={policiesUnavailable}
            onSelectionChange={onSelectedPolicyChange}
            isDisabled={isPolicyDisabled}
            trainingDevice={activeDevice}
        />
        <PolicyAccessAlert policy={selectedPolicy} />
    </Flex>
);
