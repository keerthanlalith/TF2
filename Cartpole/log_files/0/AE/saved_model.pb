??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
|
dense1_NS/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense1_NS/kernel
u
$dense1_NS/kernel/Read/ReadVariableOpReadVariableOpdense1_NS/kernel*
_output_shapes

: *
dtype0
t
dense1_NS/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense1_NS/bias
m
"dense1_NS/bias/Read/ReadVariableOpReadVariableOpdense1_NS/bias*
_output_shapes
: *
dtype0
|
dense2_NS/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense2_NS/kernel
u
$dense2_NS/kernel/Read/ReadVariableOpReadVariableOpdense2_NS/kernel*
_output_shapes

:  *
dtype0
t
dense2_NS/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense2_NS/bias
m
"dense2_NS/bias/Read/ReadVariableOpReadVariableOpdense2_NS/bias*
_output_shapes
: *
dtype0
|
dense3_NS/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense3_NS/kernel
u
$dense3_NS/kernel/Read/ReadVariableOpReadVariableOpdense3_NS/kernel*
_output_shapes

: *
dtype0
t
dense3_NS/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense3_NS/bias
m
"dense3_NS/bias/Read/ReadVariableOpReadVariableOpdense3_NS/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
 
*
0
1
2
3
!4
"5
 
*
0
1
2
3
!4
"5
?
	variables
'layer_regularization_losses
(non_trainable_variables
)metrics
	regularization_losses

trainable_variables

*layers
+layer_metrics
 
\Z
VARIABLE_VALUEdense1_NS/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense1_NS/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
,layer_regularization_losses
-non_trainable_variables
.metrics
regularization_losses
trainable_variables

/layers
0layer_metrics
 
 
 
?
	variables
1layer_regularization_losses
2non_trainable_variables
3metrics
regularization_losses
trainable_variables

4layers
5layer_metrics
\Z
VARIABLE_VALUEdense2_NS/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense2_NS/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
6layer_regularization_losses
7non_trainable_variables
8metrics
regularization_losses
trainable_variables

9layers
:layer_metrics
 
 
 
?
	variables
;layer_regularization_losses
<non_trainable_variables
=metrics
regularization_losses
trainable_variables

>layers
?layer_metrics
\Z
VARIABLE_VALUEdense3_NS/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense3_NS/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
?
#	variables
@layer_regularization_losses
Anon_trainable_variables
Bmetrics
$regularization_losses
%trainable_variables

Clayers
Dlayer_metrics
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
{
serving_default_AE_statePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_AE_statedense1_NS/kerneldense1_NS/biasdense2_NS/kerneldense2_NS/biasdense3_NS/kerneldense3_NS/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_46566
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense1_NS/kernel/Read/ReadVariableOp"dense1_NS/bias/Read/ReadVariableOp$dense2_NS/kernel/Read/ReadVariableOp"dense2_NS/bias/Read/ReadVariableOp$dense3_NS/kernel/Read/ReadVariableOp"dense3_NS/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_46766
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1_NS/kerneldense1_NS/biasdense2_NS/kerneldense2_NS/biasdense3_NS/kerneldense3_NS/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_46794??
?
?
=__inference_AE_layer_call_and_return_conditional_losses_46590

inputs,
(dense1_ns_matmul_readvariableop_resource-
)dense1_ns_biasadd_readvariableop_resource,
(dense2_ns_matmul_readvariableop_resource-
)dense2_ns_biasadd_readvariableop_resource,
(dense3_ns_matmul_readvariableop_resource-
)dense3_ns_biasadd_readvariableop_resource
identity?? dense1_NS/BiasAdd/ReadVariableOp?dense1_NS/MatMul/ReadVariableOp? dense2_NS/BiasAdd/ReadVariableOp?dense2_NS/MatMul/ReadVariableOp? dense3_NS/BiasAdd/ReadVariableOp?dense3_NS/MatMul/ReadVariableOp?
dense1_NS/MatMul/ReadVariableOpReadVariableOp(dense1_ns_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense1_NS/MatMul/ReadVariableOp?
dense1_NS/MatMulMatMulinputs'dense1_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1_NS/MatMul?
 dense1_NS/BiasAdd/ReadVariableOpReadVariableOp)dense1_ns_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense1_NS/BiasAdd/ReadVariableOp?
dense1_NS/BiasAddBiasAdddense1_NS/MatMul:product:0(dense1_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1_NS/BiasAdd?
LeakyRelu1_NS/LeakyRelu	LeakyReludense1_NS/BiasAdd:output:0*'
_output_shapes
:????????? 2
LeakyRelu1_NS/LeakyRelu?
dense2_NS/MatMul/ReadVariableOpReadVariableOp(dense2_ns_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
dense2_NS/MatMul/ReadVariableOp?
dense2_NS/MatMulMatMul%LeakyRelu1_NS/LeakyRelu:activations:0'dense2_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense2_NS/MatMul?
 dense2_NS/BiasAdd/ReadVariableOpReadVariableOp)dense2_ns_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense2_NS/BiasAdd/ReadVariableOp?
dense2_NS/BiasAddBiasAdddense2_NS/MatMul:product:0(dense2_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense2_NS/BiasAdd?
LeakyRelu2_NS/LeakyRelu	LeakyReludense2_NS/BiasAdd:output:0*'
_output_shapes
:????????? 2
LeakyRelu2_NS/LeakyRelu?
dense3_NS/MatMul/ReadVariableOpReadVariableOp(dense3_ns_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense3_NS/MatMul/ReadVariableOp?
dense3_NS/MatMulMatMul%LeakyRelu2_NS/LeakyRelu:activations:0'dense3_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/MatMul?
 dense3_NS/BiasAdd/ReadVariableOpReadVariableOp)dense3_ns_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense3_NS/BiasAdd/ReadVariableOp?
dense3_NS/BiasAddBiasAdddense3_NS/MatMul:product:0(dense3_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/BiasAdd?
IdentityIdentitydense3_NS/BiasAdd:output:0!^dense1_NS/BiasAdd/ReadVariableOp ^dense1_NS/MatMul/ReadVariableOp!^dense2_NS/BiasAdd/ReadVariableOp ^dense2_NS/MatMul/ReadVariableOp!^dense3_NS/BiasAdd/ReadVariableOp ^dense3_NS/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense1_NS/BiasAdd/ReadVariableOp dense1_NS/BiasAdd/ReadVariableOp2B
dense1_NS/MatMul/ReadVariableOpdense1_NS/MatMul/ReadVariableOp2D
 dense2_NS/BiasAdd/ReadVariableOp dense2_NS/BiasAdd/ReadVariableOp2B
dense2_NS/MatMul/ReadVariableOpdense2_NS/MatMul/ReadVariableOp2D
 dense3_NS/BiasAdd/ReadVariableOp dense3_NS/BiasAdd/ReadVariableOp2B
dense3_NS/MatMul/ReadVariableOpdense3_NS/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_AE_layer_call_fn_46631

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_AE_layer_call_and_return_conditional_losses_464942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense1_NS_layer_call_and_return_conditional_losses_46658

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense1_NS_layer_call_and_return_conditional_losses_46354

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_AE_layer_call_fn_46509
ae_state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallae_stateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_AE_layer_call_and_return_conditional_losses_464942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
~
)__inference_dense2_NS_layer_call_fn_46696

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense2_NS_layer_call_and_return_conditional_losses_463932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
I
-__inference_LeakyRelu2_NS_layer_call_fn_46706

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_464142
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_46672

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:????????? 2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
=__inference_AE_layer_call_and_return_conditional_losses_46532

inputs
dense1_ns_46514
dense1_ns_46516
dense2_ns_46520
dense2_ns_46522
dense3_ns_46526
dense3_ns_46528
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_ns_46514dense1_ns_46516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense1_NS_layer_call_and_return_conditional_losses_463542#
!dense1_NS/StatefulPartitionedCall?
LeakyRelu1_NS/PartitionedCallPartitionedCall*dense1_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_463752
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_46520dense2_ns_46522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense2_NS_layer_call_and_return_conditional_losses_463932#
!dense2_NS/StatefulPartitionedCall?
LeakyRelu2_NS/PartitionedCallPartitionedCall*dense2_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_464142
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_46526dense3_ns_46528*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense3_NS_layer_call_and_return_conditional_losses_464322#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
 __inference__wrapped_model_46340
ae_state/
+ae_dense1_ns_matmul_readvariableop_resource0
,ae_dense1_ns_biasadd_readvariableop_resource/
+ae_dense2_ns_matmul_readvariableop_resource0
,ae_dense2_ns_biasadd_readvariableop_resource/
+ae_dense3_ns_matmul_readvariableop_resource0
,ae_dense3_ns_biasadd_readvariableop_resource
identity??#AE/dense1_NS/BiasAdd/ReadVariableOp?"AE/dense1_NS/MatMul/ReadVariableOp?#AE/dense2_NS/BiasAdd/ReadVariableOp?"AE/dense2_NS/MatMul/ReadVariableOp?#AE/dense3_NS/BiasAdd/ReadVariableOp?"AE/dense3_NS/MatMul/ReadVariableOp?
"AE/dense1_NS/MatMul/ReadVariableOpReadVariableOp+ae_dense1_ns_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"AE/dense1_NS/MatMul/ReadVariableOp?
AE/dense1_NS/MatMulMatMulae_state*AE/dense1_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
AE/dense1_NS/MatMul?
#AE/dense1_NS/BiasAdd/ReadVariableOpReadVariableOp,ae_dense1_ns_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#AE/dense1_NS/BiasAdd/ReadVariableOp?
AE/dense1_NS/BiasAddBiasAddAE/dense1_NS/MatMul:product:0+AE/dense1_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
AE/dense1_NS/BiasAdd?
AE/LeakyRelu1_NS/LeakyRelu	LeakyReluAE/dense1_NS/BiasAdd:output:0*'
_output_shapes
:????????? 2
AE/LeakyRelu1_NS/LeakyRelu?
"AE/dense2_NS/MatMul/ReadVariableOpReadVariableOp+ae_dense2_ns_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"AE/dense2_NS/MatMul/ReadVariableOp?
AE/dense2_NS/MatMulMatMul(AE/LeakyRelu1_NS/LeakyRelu:activations:0*AE/dense2_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
AE/dense2_NS/MatMul?
#AE/dense2_NS/BiasAdd/ReadVariableOpReadVariableOp,ae_dense2_ns_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#AE/dense2_NS/BiasAdd/ReadVariableOp?
AE/dense2_NS/BiasAddBiasAddAE/dense2_NS/MatMul:product:0+AE/dense2_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
AE/dense2_NS/BiasAdd?
AE/LeakyRelu2_NS/LeakyRelu	LeakyReluAE/dense2_NS/BiasAdd:output:0*'
_output_shapes
:????????? 2
AE/LeakyRelu2_NS/LeakyRelu?
"AE/dense3_NS/MatMul/ReadVariableOpReadVariableOp+ae_dense3_ns_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"AE/dense3_NS/MatMul/ReadVariableOp?
AE/dense3_NS/MatMulMatMul(AE/LeakyRelu2_NS/LeakyRelu:activations:0*AE/dense3_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
AE/dense3_NS/MatMul?
#AE/dense3_NS/BiasAdd/ReadVariableOpReadVariableOp,ae_dense3_ns_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#AE/dense3_NS/BiasAdd/ReadVariableOp?
AE/dense3_NS/BiasAddBiasAddAE/dense3_NS/MatMul:product:0+AE/dense3_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
AE/dense3_NS/BiasAdd?
IdentityIdentityAE/dense3_NS/BiasAdd:output:0$^AE/dense1_NS/BiasAdd/ReadVariableOp#^AE/dense1_NS/MatMul/ReadVariableOp$^AE/dense2_NS/BiasAdd/ReadVariableOp#^AE/dense2_NS/MatMul/ReadVariableOp$^AE/dense3_NS/BiasAdd/ReadVariableOp#^AE/dense3_NS/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2J
#AE/dense1_NS/BiasAdd/ReadVariableOp#AE/dense1_NS/BiasAdd/ReadVariableOp2H
"AE/dense1_NS/MatMul/ReadVariableOp"AE/dense1_NS/MatMul/ReadVariableOp2J
#AE/dense2_NS/BiasAdd/ReadVariableOp#AE/dense2_NS/BiasAdd/ReadVariableOp2H
"AE/dense2_NS/MatMul/ReadVariableOp"AE/dense2_NS/MatMul/ReadVariableOp2J
#AE/dense3_NS/BiasAdd/ReadVariableOp#AE/dense3_NS/BiasAdd/ReadVariableOp2H
"AE/dense3_NS/MatMul/ReadVariableOp"AE/dense3_NS/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
?
=__inference_AE_layer_call_and_return_conditional_losses_46614

inputs,
(dense1_ns_matmul_readvariableop_resource-
)dense1_ns_biasadd_readvariableop_resource,
(dense2_ns_matmul_readvariableop_resource-
)dense2_ns_biasadd_readvariableop_resource,
(dense3_ns_matmul_readvariableop_resource-
)dense3_ns_biasadd_readvariableop_resource
identity?? dense1_NS/BiasAdd/ReadVariableOp?dense1_NS/MatMul/ReadVariableOp? dense2_NS/BiasAdd/ReadVariableOp?dense2_NS/MatMul/ReadVariableOp? dense3_NS/BiasAdd/ReadVariableOp?dense3_NS/MatMul/ReadVariableOp?
dense1_NS/MatMul/ReadVariableOpReadVariableOp(dense1_ns_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense1_NS/MatMul/ReadVariableOp?
dense1_NS/MatMulMatMulinputs'dense1_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1_NS/MatMul?
 dense1_NS/BiasAdd/ReadVariableOpReadVariableOp)dense1_ns_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense1_NS/BiasAdd/ReadVariableOp?
dense1_NS/BiasAddBiasAdddense1_NS/MatMul:product:0(dense1_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1_NS/BiasAdd?
LeakyRelu1_NS/LeakyRelu	LeakyReludense1_NS/BiasAdd:output:0*'
_output_shapes
:????????? 2
LeakyRelu1_NS/LeakyRelu?
dense2_NS/MatMul/ReadVariableOpReadVariableOp(dense2_ns_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
dense2_NS/MatMul/ReadVariableOp?
dense2_NS/MatMulMatMul%LeakyRelu1_NS/LeakyRelu:activations:0'dense2_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense2_NS/MatMul?
 dense2_NS/BiasAdd/ReadVariableOpReadVariableOp)dense2_ns_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense2_NS/BiasAdd/ReadVariableOp?
dense2_NS/BiasAddBiasAdddense2_NS/MatMul:product:0(dense2_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense2_NS/BiasAdd?
LeakyRelu2_NS/LeakyRelu	LeakyReludense2_NS/BiasAdd:output:0*'
_output_shapes
:????????? 2
LeakyRelu2_NS/LeakyRelu?
dense3_NS/MatMul/ReadVariableOpReadVariableOp(dense3_ns_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
dense3_NS/MatMul/ReadVariableOp?
dense3_NS/MatMulMatMul%LeakyRelu2_NS/LeakyRelu:activations:0'dense3_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/MatMul?
 dense3_NS/BiasAdd/ReadVariableOpReadVariableOp)dense3_ns_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense3_NS/BiasAdd/ReadVariableOp?
dense3_NS/BiasAddBiasAdddense3_NS/MatMul:product:0(dense3_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/BiasAdd?
IdentityIdentitydense3_NS/BiasAdd:output:0!^dense1_NS/BiasAdd/ReadVariableOp ^dense1_NS/MatMul/ReadVariableOp!^dense2_NS/BiasAdd/ReadVariableOp ^dense2_NS/MatMul/ReadVariableOp!^dense3_NS/BiasAdd/ReadVariableOp ^dense3_NS/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense1_NS/BiasAdd/ReadVariableOp dense1_NS/BiasAdd/ReadVariableOp2B
dense1_NS/MatMul/ReadVariableOpdense1_NS/MatMul/ReadVariableOp2D
 dense2_NS/BiasAdd/ReadVariableOp dense2_NS/BiasAdd/ReadVariableOp2B
dense2_NS/MatMul/ReadVariableOpdense2_NS/MatMul/ReadVariableOp2D
 dense3_NS/BiasAdd/ReadVariableOp dense3_NS/BiasAdd/ReadVariableOp2B
dense3_NS/MatMul/ReadVariableOpdense3_NS/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense3_NS_layer_call_and_return_conditional_losses_46716

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense2_NS_layer_call_and_return_conditional_losses_46393

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
=__inference_AE_layer_call_and_return_conditional_losses_46470
ae_state
dense1_ns_46452
dense1_ns_46454
dense2_ns_46458
dense2_ns_46460
dense3_ns_46464
dense3_ns_46466
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallae_statedense1_ns_46452dense1_ns_46454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense1_NS_layer_call_and_return_conditional_losses_463542#
!dense1_NS/StatefulPartitionedCall?
LeakyRelu1_NS/PartitionedCallPartitionedCall*dense1_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_463752
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_46458dense2_ns_46460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense2_NS_layer_call_and_return_conditional_losses_463932#
!dense2_NS/StatefulPartitionedCall?
LeakyRelu2_NS/PartitionedCallPartitionedCall*dense2_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_464142
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_46464dense3_ns_46466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense3_NS_layer_call_and_return_conditional_losses_464322#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
?
=__inference_AE_layer_call_and_return_conditional_losses_46449
ae_state
dense1_ns_46365
dense1_ns_46367
dense2_ns_46404
dense2_ns_46406
dense3_ns_46443
dense3_ns_46445
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallae_statedense1_ns_46365dense1_ns_46367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense1_NS_layer_call_and_return_conditional_losses_463542#
!dense1_NS/StatefulPartitionedCall?
LeakyRelu1_NS/PartitionedCallPartitionedCall*dense1_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_463752
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_46404dense2_ns_46406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense2_NS_layer_call_and_return_conditional_losses_463932#
!dense2_NS/StatefulPartitionedCall?
LeakyRelu2_NS/PartitionedCallPartitionedCall*dense2_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_464142
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_46443dense3_ns_46445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense3_NS_layer_call_and_return_conditional_losses_464322#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
?
!__inference__traced_restore_46794
file_prefix%
!assignvariableop_dense1_ns_kernel%
!assignvariableop_1_dense1_ns_bias'
#assignvariableop_2_dense2_ns_kernel%
!assignvariableop_3_dense2_ns_bias'
#assignvariableop_4_dense3_ns_kernel%
!assignvariableop_5_dense3_ns_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense1_ns_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense1_ns_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense2_ns_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense2_ns_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense3_ns_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense3_ns_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
d
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_46414

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:????????? 2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense1_NS_layer_call_fn_46667

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense1_NS_layer_call_and_return_conditional_losses_463542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_46766
file_prefix/
+savev2_dense1_ns_kernel_read_readvariableop-
)savev2_dense1_ns_bias_read_readvariableop/
+savev2_dense2_ns_kernel_read_readvariableop-
)savev2_dense2_ns_bias_read_readvariableop/
+savev2_dense3_ns_kernel_read_readvariableop-
)savev2_dense3_ns_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense1_ns_kernel_read_readvariableop)savev2_dense1_ns_bias_read_readvariableop+savev2_dense2_ns_kernel_read_readvariableop)savev2_dense2_ns_bias_read_readvariableop+savev2_dense3_ns_kernel_read_readvariableop)savev2_dense3_ns_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: : : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?	
?
D__inference_dense3_NS_layer_call_and_return_conditional_losses_46432

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense2_NS_layer_call_and_return_conditional_losses_46687

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
"__inference_AE_layer_call_fn_46648

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_AE_layer_call_and_return_conditional_losses_465322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_AE_layer_call_fn_46547
ae_state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallae_stateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_AE_layer_call_and_return_conditional_losses_465322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
?
#__inference_signature_wrapper_46566
ae_state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallae_stateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_463402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
~
)__inference_dense3_NS_layer_call_fn_46725

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense3_NS_layer_call_and_return_conditional_losses_464322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
I
-__inference_LeakyRelu1_NS_layer_call_fn_46677

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_463752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_46701

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:????????? 2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
=__inference_AE_layer_call_and_return_conditional_losses_46494

inputs
dense1_ns_46476
dense1_ns_46478
dense2_ns_46482
dense2_ns_46484
dense3_ns_46488
dense3_ns_46490
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_ns_46476dense1_ns_46478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense1_NS_layer_call_and_return_conditional_losses_463542#
!dense1_NS/StatefulPartitionedCall?
LeakyRelu1_NS/PartitionedCallPartitionedCall*dense1_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_463752
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_46482dense2_ns_46484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense2_NS_layer_call_and_return_conditional_losses_463932#
!dense2_NS/StatefulPartitionedCall?
LeakyRelu2_NS/PartitionedCallPartitionedCall*dense2_NS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_464142
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_46488dense3_ns_46490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense3_NS_layer_call_and_return_conditional_losses_464322#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_46375

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:????????? 2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
AE_state1
serving_default_AE_state:0?????????=
	dense3_NS0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?,
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
*E&call_and_return_all_conditional_losses
F__call__
G_default_save_signature"?*
_tf_keras_network?){"class_name": "Functional", "name": "AE", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "AE", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "AE_state"}, "name": "AE_state", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense1_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1_NS", "inbound_nodes": [[["AE_state", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu1_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu1_NS", "inbound_nodes": [[["dense1_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2_NS", "inbound_nodes": [[["LeakyRelu1_NS", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu2_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu2_NS", "inbound_nodes": [[["dense2_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3_NS", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3_NS", "inbound_nodes": [[["LeakyRelu2_NS", 0, 0, {}]]]}], "input_layers": [["AE_state", 0, 0]], "output_layers": [["dense3_NS", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 4]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "AE", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "AE_state"}, "name": "AE_state", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense1_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1_NS", "inbound_nodes": [[["AE_state", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu1_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu1_NS", "inbound_nodes": [[["dense1_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2_NS", "inbound_nodes": [[["LeakyRelu1_NS", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu2_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu2_NS", "inbound_nodes": [[["dense2_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3_NS", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3_NS", "inbound_nodes": [[["LeakyRelu2_NS", 0, 0, {}]]]}], "input_layers": [["AE_state", 0, 0]], "output_layers": [["dense3_NS", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mse"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.00015, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "AE_state", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "AE_state"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "LeakyRelu1_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "LeakyRelu1_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
	variables
regularization_losses
trainable_variables
 	keras_api
*N&call_and_return_all_conditional_losses
O__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "LeakyRelu2_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "LeakyRelu2_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense3_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense3_NS", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
"
	optimizer
J
0
1
2
3
!4
"5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
!4
"5"
trackable_list_wrapper
?
	variables
'layer_regularization_losses
(non_trainable_variables
)metrics
	regularization_losses

trainable_variables

*layers
+layer_metrics
F__call__
G_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Rserving_default"
signature_map
":  2dense1_NS/kernel
: 2dense1_NS/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
,layer_regularization_losses
-non_trainable_variables
.metrics
regularization_losses
trainable_variables

/layers
0layer_metrics
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
1layer_regularization_losses
2non_trainable_variables
3metrics
regularization_losses
trainable_variables

4layers
5layer_metrics
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
":   2dense2_NS/kernel
: 2dense2_NS/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
6layer_regularization_losses
7non_trainable_variables
8metrics
regularization_losses
trainable_variables

9layers
:layer_metrics
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
;layer_regularization_losses
<non_trainable_variables
=metrics
regularization_losses
trainable_variables

>layers
?layer_metrics
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
":  2dense3_NS/kernel
:2dense3_NS/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
#	variables
@layer_regularization_losses
Anon_trainable_variables
Bmetrics
$regularization_losses
%trainable_variables

Clayers
Dlayer_metrics
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
=__inference_AE_layer_call_and_return_conditional_losses_46590
=__inference_AE_layer_call_and_return_conditional_losses_46449
=__inference_AE_layer_call_and_return_conditional_losses_46614
=__inference_AE_layer_call_and_return_conditional_losses_46470?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference_AE_layer_call_fn_46631
"__inference_AE_layer_call_fn_46547
"__inference_AE_layer_call_fn_46648
"__inference_AE_layer_call_fn_46509?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_46340?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
AE_state?????????
?2?
D__inference_dense1_NS_layer_call_and_return_conditional_losses_46658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense1_NS_layer_call_fn_46667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_46672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_LeakyRelu1_NS_layer_call_fn_46677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense2_NS_layer_call_and_return_conditional_losses_46687?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense2_NS_layer_call_fn_46696?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_46701?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_LeakyRelu2_NS_layer_call_fn_46706?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense3_NS_layer_call_and_return_conditional_losses_46716?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense3_NS_layer_call_fn_46725?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_46566AE_state"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
=__inference_AE_layer_call_and_return_conditional_losses_46449j!"9?6
/?,
"?
AE_state?????????
p

 
? "%?"
?
0?????????
? ?
=__inference_AE_layer_call_and_return_conditional_losses_46470j!"9?6
/?,
"?
AE_state?????????
p 

 
? "%?"
?
0?????????
? ?
=__inference_AE_layer_call_and_return_conditional_losses_46590h!"7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
=__inference_AE_layer_call_and_return_conditional_losses_46614h!"7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
"__inference_AE_layer_call_fn_46509]!"9?6
/?,
"?
AE_state?????????
p

 
? "???????????
"__inference_AE_layer_call_fn_46547]!"9?6
/?,
"?
AE_state?????????
p 

 
? "???????????
"__inference_AE_layer_call_fn_46631[!"7?4
-?*
 ?
inputs?????????
p

 
? "???????????
"__inference_AE_layer_call_fn_46648[!"7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
H__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_46672X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? |
-__inference_LeakyRelu1_NS_layer_call_fn_46677K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
H__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_46701X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? |
-__inference_LeakyRelu2_NS_layer_call_fn_46706K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
 __inference__wrapped_model_46340r!"1?.
'?$
"?
AE_state?????????
? "5?2
0
	dense3_NS#? 
	dense3_NS??????????
D__inference_dense1_NS_layer_call_and_return_conditional_losses_46658\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? |
)__inference_dense1_NS_layer_call_fn_46667O/?,
%?"
 ?
inputs?????????
? "?????????? ?
D__inference_dense2_NS_layer_call_and_return_conditional_losses_46687\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? |
)__inference_dense2_NS_layer_call_fn_46696O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
D__inference_dense3_NS_layer_call_and_return_conditional_losses_46716\!"/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense3_NS_layer_call_fn_46725O!"/?,
%?"
 ?
inputs????????? 
? "???????????
#__inference_signature_wrapper_46566~!"=?:
? 
3?0
.
AE_state"?
AE_state?????????"5?2
0
	dense3_NS#? 
	dense3_NS?????????