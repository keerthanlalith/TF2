??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8Ų
|
dense1_NS/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense1_NS/kernel
u
$dense1_NS/kernel/Read/ReadVariableOpReadVariableOpdense1_NS/kernel*
_output_shapes

: *
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
: *!
shared_namedense3_NS/kernel
u
$dense3_NS/kernel/Read/ReadVariableOpReadVariableOpdense3_NS/kernel*
_output_shapes

: *
dtype0
t
dense3_NS/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense3_NS/bias
m
"dense3_NS/bias/Read/ReadVariableOpReadVariableOpdense3_NS/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
RMSprop/dense1_NS/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameRMSprop/dense1_NS/kernel/rms
?
0RMSprop/dense1_NS/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense1_NS/kernel/rms*
_output_shapes

: *
dtype0
?
RMSprop/dense1_NS/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/dense1_NS/bias/rms
?
.RMSprop/dense1_NS/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense1_NS/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/dense2_NS/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *-
shared_nameRMSprop/dense2_NS/kernel/rms
?
0RMSprop/dense2_NS/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense2_NS/kernel/rms*
_output_shapes

:  *
dtype0
?
RMSprop/dense2_NS/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/dense2_NS/bias/rms
?
.RMSprop/dense2_NS/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense2_NS/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/dense3_NS/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameRMSprop/dense3_NS/kernel/rms
?
0RMSprop/dense3_NS/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense3_NS/kernel/rms*
_output_shapes

: *
dtype0
?
RMSprop/dense3_NS/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/dense3_NS/bias/rms
?
.RMSprop/dense3_NS/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense3_NS/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
?$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?#
value?#B?# B?#
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
trainable_variables
		variables

regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
 	keras_api
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?
'iter
	(decay
)learning_rate
*momentum
+rho	rmsU	rmsV	rmsW	rmsX	!rmsY	"rmsZ
*
0
1
2
3
!4
"5
*
0
1
2
3
!4
"5
 
?
,layer_regularization_losses
trainable_variables
		variables
-non_trainable_variables
.metrics

regularization_losses
/layer_metrics

0layers
 
\Z
VARIABLE_VALUEdense1_NS/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense1_NS/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
1layer_regularization_losses
trainable_variables
	variables
2non_trainable_variables
3metrics
regularization_losses
4layer_metrics

5layers
 
 
 
?
6layer_regularization_losses
trainable_variables
	variables
7non_trainable_variables
8metrics
regularization_losses
9layer_metrics

:layers
\Z
VARIABLE_VALUEdense2_NS/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense2_NS/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
;layer_regularization_losses
trainable_variables
	variables
<non_trainable_variables
=metrics
regularization_losses
>layer_metrics

?layers
 
 
 
?
@layer_regularization_losses
trainable_variables
	variables
Anon_trainable_variables
Bmetrics
regularization_losses
Clayer_metrics

Dlayers
\Z
VARIABLE_VALUEdense3_NS/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense3_NS/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
Elayer_regularization_losses
#trainable_variables
$	variables
Fnon_trainable_variables
Gmetrics
%regularization_losses
Hlayer_metrics

Ilayers
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
 

J0
K1
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
4
	Ltotal
	Mcount
N	variables
O	keras_api
D
	Ptotal
	Qcount
R
_fn_kwargs
S	variables
T	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

N	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

S	variables
??
VARIABLE_VALUERMSprop/dense1_NS/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense1_NS/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense2_NS/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense2_NS/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense3_NS/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense3_NS/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_AE_statePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_AE_statedense1_NS/kerneldense1_NS/biasdense2_NS/kerneldense2_NS/biasdense3_NS/kerneldense3_NS/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_4835502
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense1_NS/kernel/Read/ReadVariableOp"dense1_NS/bias/Read/ReadVariableOp$dense2_NS/kernel/Read/ReadVariableOp"dense2_NS/bias/Read/ReadVariableOp$dense3_NS/kernel/Read/ReadVariableOp"dense3_NS/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0RMSprop/dense1_NS/kernel/rms/Read/ReadVariableOp.RMSprop/dense1_NS/bias/rms/Read/ReadVariableOp0RMSprop/dense2_NS/kernel/rms/Read/ReadVariableOp.RMSprop/dense2_NS/bias/rms/Read/ReadVariableOp0RMSprop/dense3_NS/kernel/rms/Read/ReadVariableOp.RMSprop/dense3_NS/bias/rms/Read/ReadVariableOpConst*"
Tin
2	*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_4835747
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1_NS/kerneldense1_NS/biasdense2_NS/kerneldense2_NS/biasdense3_NS/kerneldense3_NS/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/dense1_NS/kernel/rmsRMSprop/dense1_NS/bias/rmsRMSprop/dense2_NS/kernel/rmsRMSprop/dense2_NS/bias/rmsRMSprop/dense3_NS/kernel/rmsRMSprop/dense3_NS/bias/rms*!
Tin
2*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_4835820??
?
?
+__inference_dense3_NS_layer_call_fn_4835661

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense3_NS_layer_call_and_return_conditional_losses_48353602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?Z
?

#__inference__traced_restore_4835820
file_prefix%
!assignvariableop_dense1_ns_kernel%
!assignvariableop_1_dense1_ns_bias'
#assignvariableop_2_dense2_ns_kernel%
!assignvariableop_3_dense2_ns_bias'
#assignvariableop_4_dense3_ns_kernel%
!assignvariableop_5_dense3_ns_bias#
assignvariableop_6_rmsprop_iter$
 assignvariableop_7_rmsprop_decay,
(assignvariableop_8_rmsprop_learning_rate'
#assignvariableop_9_rmsprop_momentum#
assignvariableop_10_rmsprop_rho
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_14
0assignvariableop_15_rmsprop_dense1_ns_kernel_rms2
.assignvariableop_16_rmsprop_dense1_ns_bias_rms4
0assignvariableop_17_rmsprop_dense2_ns_kernel_rms2
.assignvariableop_18_rmsprop_dense2_ns_bias_rms4
0assignvariableop_19_rmsprop_dense3_ns_kernel_rms2
.assignvariableop_20_rmsprop_dense3_ns_bias_rms
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
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
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_rmsprop_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_rmsprop_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rmsprop_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_rmsprop_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_rmsprop_dense1_ns_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp.assignvariableop_16_rmsprop_dense1_ns_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_rmsprop_dense2_ns_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_rmsprop_dense2_ns_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp0assignvariableop_19_rmsprop_dense3_ns_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp.assignvariableop_20_rmsprop_dense3_ns_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21?
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
?__inference_AE_layer_call_and_return_conditional_losses_4835398
ae_state
dense1_ns_4835380
dense1_ns_4835382
dense2_ns_4835386
dense2_ns_4835388
dense3_ns_4835392
dense3_ns_4835394
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallae_statedense1_ns_4835380dense1_ns_4835382*
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
GPU 2J 8? *O
fJRH
F__inference_dense1_NS_layer_call_and_return_conditional_losses_48352822#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_48353032
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_4835386dense2_ns_4835388*
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
GPU 2J 8? *O
fJRH
F__inference_dense2_NS_layer_call_and_return_conditional_losses_48353212#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_48353422
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_4835392dense3_ns_4835394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense3_NS_layer_call_and_return_conditional_losses_48353602#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
?
"__inference__wrapped_model_4835268
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

: *
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

: *
dtype02$
"AE/dense3_NS/MatMul/ReadVariableOp?
AE/dense3_NS/MatMulMatMul(AE/LeakyRelu2_NS/LeakyRelu:activations:0*AE/dense3_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
AE/dense3_NS/MatMul?
#AE/dense3_NS/BiasAdd/ReadVariableOpReadVariableOp,ae_dense3_ns_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#AE/dense3_NS/BiasAdd/ReadVariableOp?
AE/dense3_NS/BiasAddBiasAddAE/dense3_NS/MatMul:product:0+AE/dense3_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
AE/dense3_NS/BiasAdd?
IdentityIdentityAE/dense3_NS/BiasAdd:output:0$^AE/dense1_NS/BiasAdd/ReadVariableOp#^AE/dense1_NS/MatMul/ReadVariableOp$^AE/dense2_NS/BiasAdd/ReadVariableOp#^AE/dense2_NS/MatMul/ReadVariableOp$^AE/dense3_NS/BiasAdd/ReadVariableOp#^AE/dense3_NS/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2J
#AE/dense1_NS/BiasAdd/ReadVariableOp#AE/dense1_NS/BiasAdd/ReadVariableOp2H
"AE/dense1_NS/MatMul/ReadVariableOp"AE/dense1_NS/MatMul/ReadVariableOp2J
#AE/dense2_NS/BiasAdd/ReadVariableOp#AE/dense2_NS/BiasAdd/ReadVariableOp2H
"AE/dense2_NS/MatMul/ReadVariableOp"AE/dense2_NS/MatMul/ReadVariableOp2J
#AE/dense3_NS/BiasAdd/ReadVariableOp#AE/dense3_NS/BiasAdd/ReadVariableOp2H
"AE/dense3_NS/MatMul/ReadVariableOp"AE/dense3_NS/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?	
?
F__inference_dense2_NS_layer_call_and_return_conditional_losses_4835321

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
?3
?
 __inference__traced_save_4835747
file_prefix/
+savev2_dense1_ns_kernel_read_readvariableop-
)savev2_dense1_ns_bias_read_readvariableop/
+savev2_dense2_ns_kernel_read_readvariableop-
)savev2_dense2_ns_bias_read_readvariableop/
+savev2_dense3_ns_kernel_read_readvariableop-
)savev2_dense3_ns_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_rmsprop_dense1_ns_kernel_rms_read_readvariableop9
5savev2_rmsprop_dense1_ns_bias_rms_read_readvariableop;
7savev2_rmsprop_dense2_ns_kernel_rms_read_readvariableop9
5savev2_rmsprop_dense2_ns_bias_rms_read_readvariableop;
7savev2_rmsprop_dense3_ns_kernel_rms_read_readvariableop9
5savev2_rmsprop_dense3_ns_bias_rms_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense1_ns_kernel_read_readvariableop)savev2_dense1_ns_bias_read_readvariableop+savev2_dense2_ns_kernel_read_readvariableop)savev2_dense2_ns_bias_read_readvariableop+savev2_dense3_ns_kernel_read_readvariableop)savev2_dense3_ns_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_rmsprop_dense1_ns_kernel_rms_read_readvariableop5savev2_rmsprop_dense1_ns_bias_rms_read_readvariableop7savev2_rmsprop_dense2_ns_kernel_rms_read_readvariableop5savev2_rmsprop_dense2_ns_bias_rms_read_readvariableop7savev2_rmsprop_dense3_ns_kernel_rms_read_readvariableop5savev2_rmsprop_dense3_ns_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapesx
v: : : :  : : :: : : : : : : : : : : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?
?
?__inference_AE_layer_call_and_return_conditional_losses_4835550

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

: *
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

: *
dtype02!
dense3_NS/MatMul/ReadVariableOp?
dense3_NS/MatMulMatMul%LeakyRelu2_NS/LeakyRelu:activations:0'dense3_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/MatMul?
 dense3_NS/BiasAdd/ReadVariableOpReadVariableOp)dense3_ns_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense3_NS/BiasAdd/ReadVariableOp?
dense3_NS/BiasAddBiasAdddense3_NS/MatMul:product:0(dense3_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/BiasAdd?
IdentityIdentitydense3_NS/BiasAdd:output:0!^dense1_NS/BiasAdd/ReadVariableOp ^dense1_NS/MatMul/ReadVariableOp!^dense2_NS/BiasAdd/ReadVariableOp ^dense2_NS/MatMul/ReadVariableOp!^dense3_NS/BiasAdd/ReadVariableOp ^dense3_NS/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense1_NS/BiasAdd/ReadVariableOp dense1_NS/BiasAdd/ReadVariableOp2B
dense1_NS/MatMul/ReadVariableOpdense1_NS/MatMul/ReadVariableOp2D
 dense2_NS/BiasAdd/ReadVariableOp dense2_NS/BiasAdd/ReadVariableOp2B
dense2_NS/MatMul/ReadVariableOpdense2_NS/MatMul/ReadVariableOp2D
 dense3_NS/BiasAdd/ReadVariableOp dense3_NS/BiasAdd/ReadVariableOp2B
dense3_NS/MatMul/ReadVariableOpdense3_NS/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense3_NS_layer_call_and_return_conditional_losses_4835652

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
f
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_4835342

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
?	
?
F__inference_dense1_NS_layer_call_and_return_conditional_losses_4835282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_4835637

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
?
K
/__inference_LeakyRelu1_NS_layer_call_fn_4835613

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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_48353032
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
?
?
$__inference_AE_layer_call_fn_4835584

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
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_AE_layer_call_and_return_conditional_losses_48354602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense3_NS_layer_call_and_return_conditional_losses_4835360

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
?
?__inference_AE_layer_call_and_return_conditional_losses_4835377
ae_state
dense1_ns_4835293
dense1_ns_4835295
dense2_ns_4835332
dense2_ns_4835334
dense3_ns_4835371
dense3_ns_4835373
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallae_statedense1_ns_4835293dense1_ns_4835295*
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
GPU 2J 8? *O
fJRH
F__inference_dense1_NS_layer_call_and_return_conditional_losses_48352822#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_48353032
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_4835332dense2_ns_4835334*
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
GPU 2J 8? *O
fJRH
F__inference_dense2_NS_layer_call_and_return_conditional_losses_48353212#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_48353422
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_4835371dense3_ns_4835373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense3_NS_layer_call_and_return_conditional_losses_48353602#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
?
?__inference_AE_layer_call_and_return_conditional_losses_4835460

inputs
dense1_ns_4835442
dense1_ns_4835444
dense2_ns_4835448
dense2_ns_4835450
dense3_ns_4835454
dense3_ns_4835456
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_ns_4835442dense1_ns_4835444*
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
GPU 2J 8? *O
fJRH
F__inference_dense1_NS_layer_call_and_return_conditional_losses_48352822#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_48353032
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_4835448dense2_ns_4835450*
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
GPU 2J 8? *O
fJRH
F__inference_dense2_NS_layer_call_and_return_conditional_losses_48353212#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_48353422
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_4835454dense3_ns_4835456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense3_NS_layer_call_and_return_conditional_losses_48353602#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_AE_layer_call_fn_4835437
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
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_AE_layer_call_and_return_conditional_losses_48354222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
f
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_4835608

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
?
f
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_4835303

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
?	
?
F__inference_dense2_NS_layer_call_and_return_conditional_losses_4835623

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
?
?
?__inference_AE_layer_call_and_return_conditional_losses_4835526

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

: *
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

: *
dtype02!
dense3_NS/MatMul/ReadVariableOp?
dense3_NS/MatMulMatMul%LeakyRelu2_NS/LeakyRelu:activations:0'dense3_NS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/MatMul?
 dense3_NS/BiasAdd/ReadVariableOpReadVariableOp)dense3_ns_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense3_NS/BiasAdd/ReadVariableOp?
dense3_NS/BiasAddBiasAdddense3_NS/MatMul:product:0(dense3_NS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense3_NS/BiasAdd?
IdentityIdentitydense3_NS/BiasAdd:output:0!^dense1_NS/BiasAdd/ReadVariableOp ^dense1_NS/MatMul/ReadVariableOp!^dense2_NS/BiasAdd/ReadVariableOp ^dense2_NS/MatMul/ReadVariableOp!^dense3_NS/BiasAdd/ReadVariableOp ^dense3_NS/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense1_NS/BiasAdd/ReadVariableOp dense1_NS/BiasAdd/ReadVariableOp2B
dense1_NS/MatMul/ReadVariableOpdense1_NS/MatMul/ReadVariableOp2D
 dense2_NS/BiasAdd/ReadVariableOp dense2_NS/BiasAdd/ReadVariableOp2B
dense2_NS/MatMul/ReadVariableOpdense2_NS/MatMul/ReadVariableOp2D
 dense3_NS/BiasAdd/ReadVariableOp dense3_NS/BiasAdd/ReadVariableOp2B
dense3_NS/MatMul/ReadVariableOpdense3_NS/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense2_NS_layer_call_fn_4835632

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
GPU 2J 8? *O
fJRH
F__inference_dense2_NS_layer_call_and_return_conditional_losses_48353212
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
?
?
%__inference_signature_wrapper_4835502
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
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_48352682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
K
/__inference_LeakyRelu2_NS_layer_call_fn_4835642

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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_48353422
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
?
?
$__inference_AE_layer_call_fn_4835475
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
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_AE_layer_call_and_return_conditional_losses_48354602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
AE_state
?
?
+__inference_dense1_NS_layer_call_fn_4835603

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
GPU 2J 8? *O
fJRH
F__inference_dense1_NS_layer_call_and_return_conditional_losses_48352822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_AE_layer_call_and_return_conditional_losses_4835422

inputs
dense1_ns_4835404
dense1_ns_4835406
dense2_ns_4835410
dense2_ns_4835412
dense3_ns_4835416
dense3_ns_4835418
identity??!dense1_NS/StatefulPartitionedCall?!dense2_NS/StatefulPartitionedCall?!dense3_NS/StatefulPartitionedCall?
!dense1_NS/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_ns_4835404dense1_ns_4835406*
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
GPU 2J 8? *O
fJRH
F__inference_dense1_NS_layer_call_and_return_conditional_losses_48352822#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_48353032
LeakyRelu1_NS/PartitionedCall?
!dense2_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu1_NS/PartitionedCall:output:0dense2_ns_4835410dense2_ns_4835412*
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
GPU 2J 8? *O
fJRH
F__inference_dense2_NS_layer_call_and_return_conditional_losses_48353212#
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
GPU 2J 8? *S
fNRL
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_48353422
LeakyRelu2_NS/PartitionedCall?
!dense3_NS/StatefulPartitionedCallStatefulPartitionedCall&LeakyRelu2_NS/PartitionedCall:output:0dense3_ns_4835416dense3_ns_4835418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense3_NS_layer_call_and_return_conditional_losses_48353602#
!dense3_NS/StatefulPartitionedCall?
IdentityIdentity*dense3_NS/StatefulPartitionedCall:output:0"^dense1_NS/StatefulPartitionedCall"^dense2_NS/StatefulPartitionedCall"^dense3_NS/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense1_NS/StatefulPartitionedCall!dense1_NS/StatefulPartitionedCall2F
!dense2_NS/StatefulPartitionedCall!dense2_NS/StatefulPartitionedCall2F
!dense3_NS/StatefulPartitionedCall!dense3_NS/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense1_NS_layer_call_and_return_conditional_losses_4835594

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_AE_layer_call_fn_4835567

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
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_AE_layer_call_and_return_conditional_losses_48354222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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
serving_default_AE_state:0?????????=
	dense3_NS0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ь
?-
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
trainable_variables
		variables

regularization_losses
	keras_api

signatures
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses"?+
_tf_keras_network?*{"class_name": "Functional", "name": "AE", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "AE", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "AE_state"}, "name": "AE_state", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense1_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1_NS", "inbound_nodes": [[["AE_state", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu1_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu1_NS", "inbound_nodes": [[["dense1_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2_NS", "inbound_nodes": [[["LeakyRelu1_NS", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu2_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu2_NS", "inbound_nodes": [[["dense2_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3_NS", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3_NS", "inbound_nodes": [[["LeakyRelu2_NS", 0, 0, {}]]]}], "input_layers": [["AE_state", 0, 0]], "output_layers": [["dense3_NS", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "AE", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "AE_state"}, "name": "AE_state", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense1_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1_NS", "inbound_nodes": [[["AE_state", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu1_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu1_NS", "inbound_nodes": [[["dense1_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2_NS", "inbound_nodes": [[["LeakyRelu1_NS", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "LeakyRelu2_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "LeakyRelu2_NS", "inbound_nodes": [[["dense2_NS", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3_NS", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3_NS", "inbound_nodes": [[["LeakyRelu2_NS", 0, 0, {}]]]}], "input_layers": [["AE_state", 0, 0]], "output_layers": [["dense3_NS", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0001500000071246177, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "AE_state", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "AE_state"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "LeakyRelu1_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "LeakyRelu1_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
b__call__
*c&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2_NS", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
trainable_variables
	variables
regularization_losses
 	keras_api
d__call__
*e&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "LeakyRelu2_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "LeakyRelu2_NS", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
f__call__
*g&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense3_NS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense3_NS", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
'iter
	(decay
)learning_rate
*momentum
+rho	rmsU	rmsV	rmsW	rmsX	!rmsY	"rmsZ"
	optimizer
J
0
1
2
3
!4
"5"
trackable_list_wrapper
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
?
,layer_regularization_losses
trainable_variables
		variables
-non_trainable_variables
.metrics

regularization_losses
/layer_metrics

0layers
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
hserving_default"
signature_map
":  2dense1_NS/kernel
: 2dense1_NS/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1layer_regularization_losses
trainable_variables
	variables
2non_trainable_variables
3metrics
regularization_losses
4layer_metrics

5layers
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6layer_regularization_losses
trainable_variables
	variables
7non_trainable_variables
8metrics
regularization_losses
9layer_metrics

:layers
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
":   2dense2_NS/kernel
: 2dense2_NS/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;layer_regularization_losses
trainable_variables
	variables
<non_trainable_variables
=metrics
regularization_losses
>layer_metrics

?layers
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@layer_regularization_losses
trainable_variables
	variables
Anon_trainable_variables
Bmetrics
regularization_losses
Clayer_metrics

Dlayers
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
":  2dense3_NS/kernel
:2dense3_NS/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Elayer_regularization_losses
#trainable_variables
$	variables
Fnon_trainable_variables
Gmetrics
%regularization_losses
Hlayer_metrics

Ilayers
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
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
 "
trackable_list_wrapper
?
	Ltotal
	Mcount
N	variables
O	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Ptotal
	Qcount
R
_fn_kwargs
S	variables
T	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
-
S	variables"
_generic_user_object
,:* 2RMSprop/dense1_NS/kernel/rms
&:$ 2RMSprop/dense1_NS/bias/rms
,:*  2RMSprop/dense2_NS/kernel/rms
&:$ 2RMSprop/dense2_NS/bias/rms
,:* 2RMSprop/dense3_NS/kernel/rms
&:$2RMSprop/dense3_NS/bias/rms
?2?
$__inference_AE_layer_call_fn_4835475
$__inference_AE_layer_call_fn_4835567
$__inference_AE_layer_call_fn_4835437
$__inference_AE_layer_call_fn_4835584?
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
"__inference__wrapped_model_4835268?
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
AE_state?????????
?2?
?__inference_AE_layer_call_and_return_conditional_losses_4835526
?__inference_AE_layer_call_and_return_conditional_losses_4835550
?__inference_AE_layer_call_and_return_conditional_losses_4835377
?__inference_AE_layer_call_and_return_conditional_losses_4835398?
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
+__inference_dense1_NS_layer_call_fn_4835603?
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
F__inference_dense1_NS_layer_call_and_return_conditional_losses_4835594?
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
/__inference_LeakyRelu1_NS_layer_call_fn_4835613?
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
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_4835608?
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
+__inference_dense2_NS_layer_call_fn_4835632?
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
F__inference_dense2_NS_layer_call_and_return_conditional_losses_4835623?
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
/__inference_LeakyRelu2_NS_layer_call_fn_4835642?
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
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_4835637?
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
+__inference_dense3_NS_layer_call_fn_4835661?
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
F__inference_dense3_NS_layer_call_and_return_conditional_losses_4835652?
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
%__inference_signature_wrapper_4835502AE_state"?
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
?__inference_AE_layer_call_and_return_conditional_losses_4835377j!"9?6
/?,
"?
AE_state?????????
p

 
? "%?"
?
0?????????
? ?
?__inference_AE_layer_call_and_return_conditional_losses_4835398j!"9?6
/?,
"?
AE_state?????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_AE_layer_call_and_return_conditional_losses_4835526h!"7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
?__inference_AE_layer_call_and_return_conditional_losses_4835550h!"7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
$__inference_AE_layer_call_fn_4835437]!"9?6
/?,
"?
AE_state?????????
p

 
? "???????????
$__inference_AE_layer_call_fn_4835475]!"9?6
/?,
"?
AE_state?????????
p 

 
? "???????????
$__inference_AE_layer_call_fn_4835567[!"7?4
-?*
 ?
inputs?????????
p

 
? "???????????
$__inference_AE_layer_call_fn_4835584[!"7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
J__inference_LeakyRelu1_NS_layer_call_and_return_conditional_losses_4835608X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ~
/__inference_LeakyRelu1_NS_layer_call_fn_4835613K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
J__inference_LeakyRelu2_NS_layer_call_and_return_conditional_losses_4835637X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ~
/__inference_LeakyRelu2_NS_layer_call_fn_4835642K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
"__inference__wrapped_model_4835268r!"1?.
'?$
"?
AE_state?????????
? "5?2
0
	dense3_NS#? 
	dense3_NS??????????
F__inference_dense1_NS_layer_call_and_return_conditional_losses_4835594\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? ~
+__inference_dense1_NS_layer_call_fn_4835603O/?,
%?"
 ?
inputs?????????
? "?????????? ?
F__inference_dense2_NS_layer_call_and_return_conditional_losses_4835623\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ~
+__inference_dense2_NS_layer_call_fn_4835632O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
F__inference_dense3_NS_layer_call_and_return_conditional_losses_4835652\!"/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? ~
+__inference_dense3_NS_layer_call_fn_4835661O!"/?,
%?"
 ?
inputs????????? 
? "???????????
%__inference_signature_wrapper_4835502~!"=?:
? 
3?0
.
AE_state"?
AE_state?????????"5?2
0
	dense3_NS#? 
	dense3_NS?????????