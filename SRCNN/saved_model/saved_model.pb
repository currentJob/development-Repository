ЩВ3
ЗЏ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resourceИ
ы
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.13.12v2.13.0-17-gf841394b1b78јб*
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
В
Adam/v/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/conv2d_11/bias
{
)Adam/v/conv2d_11/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_11/bias*
_output_shapes
:*
dtype0
В
Adam/m/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/conv2d_11/bias
{
)Adam/m/conv2d_11/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_11/bias*
_output_shapes
:*
dtype0
Т
Adam/v/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/v/conv2d_11/kernel
Л
+Adam/v/conv2d_11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_11/kernel*&
_output_shapes
:@*
dtype0
Т
Adam/m/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/m/conv2d_11/kernel
Л
+Adam/m/conv2d_11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_11/kernel*&
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_10/beta
Х
6Adam/v/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_10/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_10/beta
Х
6Adam/m/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_10/beta*
_output_shapes
:@*
dtype0
Ю
#Adam/v/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/v/batch_normalization_10/gamma
Ч
7Adam/v/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_10/gamma*
_output_shapes
:@*
dtype0
Ю
#Adam/m/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/m/batch_normalization_10/gamma
Ч
7Adam/m/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_10/gamma*
_output_shapes
:@*
dtype0
В
Adam/v/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_10/bias
{
)Adam/v/conv2d_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_10/bias*
_output_shapes
:@*
dtype0
В
Adam/m/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_10/bias
{
)Adam/m/conv2d_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_10/bias*
_output_shapes
:@*
dtype0
Т
Adam/v/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/v/conv2d_10/kernel
Л
+Adam/v/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_10/kernel*&
_output_shapes
:@@*
dtype0
Т
Adam/m/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/m/conv2d_10/kernel
Л
+Adam/m/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_10/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_9/beta
У
5Adam/v/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_9/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_9/beta
У
5Adam/m/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_9/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_9/gamma
Х
6Adam/v/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_9/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_9/gamma
Х
6Adam/m/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_9/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_9/bias
y
(Adam/v/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_9/bias
y
(Adam/m/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_9/kernel
Й
*Adam/v/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_9/kernel
Й
*Adam/m/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_8/beta
У
5Adam/v/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_8/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_8/beta
У
5Adam/m/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_8/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_8/gamma
Х
6Adam/v/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_8/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_8/gamma
Х
6Adam/m/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_8/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_8/bias
y
(Adam/v/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_8/bias
y
(Adam/m/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_8/kernel
Й
*Adam/v/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_8/kernel
Й
*Adam/m/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_7/beta
У
5Adam/v/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_7/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_7/beta
У
5Adam/m/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_7/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_7/gamma
Х
6Adam/v/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_7/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_7/gamma
Х
6Adam/m/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_7/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_7/bias
y
(Adam/v/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_7/bias
y
(Adam/m/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_7/kernel
Й
*Adam/v/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_7/kernel
Й
*Adam/m/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_6/beta
У
5Adam/v/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_6/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_6/beta
У
5Adam/m/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_6/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_6/gamma
Х
6Adam/v/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_6/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_6/gamma
Х
6Adam/m/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_6/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_6/bias
y
(Adam/v/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_6/bias
y
(Adam/m/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_6/kernel
Й
*Adam/v/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_6/kernel
Й
*Adam/m/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_5/beta
У
5Adam/v/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_5/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_5/beta
У
5Adam/m/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_5/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_5/gamma
Х
6Adam/v/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_5/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_5/gamma
Х
6Adam/m/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_5/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_5/bias
y
(Adam/v/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_5/bias
y
(Adam/m/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_5/kernel
Й
*Adam/v/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_5/kernel
Й
*Adam/m/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_4/beta
У
5Adam/v/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_4/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_4/beta
У
5Adam/m/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_4/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_4/gamma
Х
6Adam/v/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_4/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_4/gamma
Х
6Adam/m/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_4/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_4/bias
y
(Adam/v/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_4/bias
y
(Adam/m/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_4/kernel
Й
*Adam/v/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_4/kernel
Й
*Adam/m/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_3/beta
У
5Adam/v/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_3/beta
У
5Adam/m/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_3/gamma
Х
6Adam/v/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_3/gamma
Х
6Adam/m/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_3/bias
y
(Adam/v/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_3/bias
y
(Adam/m/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_3/kernel
Й
*Adam/v/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_3/kernel
Й
*Adam/m/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_2/beta
У
5Adam/v/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_2/beta
У
5Adam/m/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_2/gamma
Х
6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_2/gamma
Х
6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_2/bias
y
(Adam/v/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_2/bias
y
(Adam/m/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_2/kernel
Й
*Adam/v/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_2/kernel
Й
*Adam/m/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_1/beta
У
5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_1/beta
У
5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_1/gamma
Х
6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_1/gamma
Х
6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_1/bias
y
(Adam/v/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_1/bias
y
(Adam/m/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_1/kernel
Й
*Adam/v/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0
Р
Adam/m/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_1/kernel
Й
*Adam/m/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0
Ц
Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/v/batch_normalization/beta
П
3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
:@*
dtype0
Ц
Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/m/batch_normalization/beta
П
3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
:@*
dtype0
Ш
 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/v/batch_normalization/gamma
С
4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
:@*
dtype0
Ш
 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/m/batch_normalization/gamma
С
4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
:@*
dtype0
|
Adam/v/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/v/conv2d/bias
u
&Adam/v/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/bias*
_output_shapes
:@*
dtype0
|
Adam/m/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/m/conv2d/bias
u
&Adam/m/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/bias*
_output_shapes
:@*
dtype0
М
Adam/v/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d/kernel
Е
(Adam/v/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/kernel*&
_output_shapes
:@*
dtype0
М
Adam/m/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d/kernel
Е
(Adam/m/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/kernel*&
_output_shapes
:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
Д
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:@*
dtype0
§
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_10/moving_variance
Э
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_10/moving_mean
Х
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:@*
dtype0
О
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_10/beta
З
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:@*
dtype0
Р
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_10/gamma
Й
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:@*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:@*
dtype0
Д
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_9/moving_variance
Ы
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_9/moving_mean
У
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_9/beta
Е
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_9/gamma
З
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:@*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:@*
dtype0
В
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_8/moving_variance
Ы
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_8/moving_mean
У
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_8/beta
Е
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_8/gamma
З
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0
В
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
В
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
В
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
Ѓ
serving_default_input_1Placeholder*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
у
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/bias*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_19401

NoOpNoOp
њ№
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*щџ
valueоџBкџ Bвџ
Э

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!layer_with_weights-21
!layer-32
"layer-33
#layer_with_weights-22
#layer-34
$layer-35
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_default_save_signature
,	optimizer
-
signatures*
* 
»
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op*
’
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance*
О
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
»
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
 P_jit_compiled_convolution_op*
’
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance*
О
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
»
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op*
’
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance*
О
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
Ќ
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias
!Д_jit_compiled_convolution_op*
а
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
	Лaxis

Мgamma
	Нbeta
Оmoving_mean
Пmoving_variance*
Ф
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses* 
—
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses
Ьkernel
	Эbias
!Ю_jit_compiled_convolution_op*
а
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
£__call__
+§&call_and_return_all_conditional_losses
	•axis

¶gamma
	Іbeta
®moving_mean
©moving_variance*
Ф
™	variables
Ђtrainable_variables
ђregularization_losses
≠	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses* 
—
∞	variables
±trainable_variables
≤regularization_losses
≥	keras_api
і__call__
+µ&call_and_return_all_conditional_losses
ґkernel
	Јbias
!Є_jit_compiled_convolution_op*
а
є	variables
Їtrainable_variables
їregularization_losses
Љ	keras_api
љ__call__
+Њ&call_and_return_all_conditional_losses
	њaxis

јgamma
	Ѕbeta
¬moving_mean
√moving_variance*
Ф
ƒ	variables
≈trainable_variables
∆regularization_losses
«	keras_api
»__call__
+…&call_and_return_all_conditional_losses* 
—
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses
–kernel
	—bias
!“_jit_compiled_convolution_op*
а
”	variables
‘trainable_variables
’regularization_losses
÷	keras_api
„__call__
+Ў&call_and_return_all_conditional_losses
	ўaxis

Џgamma
	џbeta
№moving_mean
Ёmoving_variance*
Ф
ё	variables
яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses* 
—
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses
кkernel
	лbias
!м_jit_compiled_convolution_op*
а
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses
	уaxis

фgamma
	хbeta
цmoving_mean
чmoving_variance*
Ф
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses* 
—
ю	variables
€trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
Дkernel
	Еbias
!Ж_jit_compiled_convolution_op*
а
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
	Нaxis

Оgamma
	Пbeta
Рmoving_mean
Сmoving_variance*
Ф
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses* 
—
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Юkernel
	Яbias
!†_jit_compiled_convolution_op*
а
°	variables
Ґtrainable_variables
£regularization_losses
§	keras_api
•__call__
+¶&call_and_return_all_conditional_losses
	Іaxis

®gamma
	©beta
™moving_mean
Ђmoving_variance*
Ф
ђ	variables
≠trainable_variables
Ѓregularization_losses
ѓ	keras_api
∞__call__
+±&call_and_return_all_conditional_losses* 
—
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses
Єkernel
	єbias
!Ї_jit_compiled_convolution_op*
а
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses
	Ѕaxis

¬gamma
	√beta
ƒmoving_mean
≈moving_variance*
Ф
∆	variables
«trainable_variables
»regularization_losses
…	keras_api
 __call__
+Ћ&call_and_return_all_conditional_losses* 
—
ћ	variables
Ќtrainable_variables
ќregularization_losses
ѕ	keras_api
–__call__
+—&call_and_return_all_conditional_losses
“kernel
	”bias
!‘_jit_compiled_convolution_op*
Ф
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses* 
ћ
40
51
>2
?3
@4
A5
N6
O7
X8
Y9
Z10
[11
h12
i13
r14
s15
t16
u17
В18
Г19
М20
Н21
О22
П23
Ь24
Э25
¶26
І27
®28
©29
ґ30
Ј31
ј32
Ѕ33
¬34
√35
–36
—37
Џ38
џ39
№40
Ё41
к42
л43
ф44
х45
ц46
ч47
Д48
Е49
О50
П51
Р52
С53
Ю54
Я55
®56
©57
™58
Ђ59
Є60
є61
¬62
√63
ƒ64
≈65
“66
”67*
М
40
51
>2
?3
N4
O5
X6
Y7
h8
i9
r10
s11
В12
Г13
М14
Н15
Ь16
Э17
¶18
І19
ґ20
Ј21
ј22
Ѕ23
–24
—25
Џ26
џ27
к28
л29
ф30
х31
Д32
Е33
О34
П35
Ю36
Я37
®38
©39
Є40
є41
¬42
√43
“44
”45*
* 
µ
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
+_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

аtrace_0
бtrace_1* 

вtrace_0
гtrace_1* 
* 
И
д
_variables
е_iterations
ж_learning_rate
з_index_dict
и
_momentums
й_velocities
к_update_step_xla*

лserving_default* 

40
51*

40
51*
* 
Ш
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

сtrace_0* 

тtrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
>0
?1
@2
A3*

>0
?1*
* 
Ш
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

шtrace_0
щtrace_1* 

ъtrace_0
ыtrace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
ьnon_trainable_variables
эlayers
юmetrics
 €layer_regularization_losses
Аlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

Бtrace_0* 

Вtrace_0* 

N0
O1*

N0
O1*
* 
Ш
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

Иtrace_0* 

Йtrace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
X0
Y1
Z2
[3*

X0
Y1*
* 
Ш
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

Пtrace_0
Рtrace_1* 

Сtrace_0
Тtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

Шtrace_0* 

Щtrace_0* 

h0
i1*

h0
i1*
* 
Ш
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

Яtrace_0* 

†trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
r0
s1
t2
u3*

r0
s1*
* 
Ш
°non_trainable_variables
Ґlayers
£metrics
 §layer_regularization_losses
•layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

¶trace_0
Іtrace_1* 

®trace_0
©trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

ѓtrace_0* 

∞trace_0* 

В0
Г1*

В0
Г1*
* 
Ы
±non_trainable_variables
≤layers
≥metrics
 іlayer_regularization_losses
µlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

ґtrace_0* 

Јtrace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
М0
Н1
О2
П3*

М0
Н1*
* 
Ю
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*

љtrace_0
Њtrace_1* 

њtrace_0
јtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Ѕnon_trainable_variables
¬layers
√metrics
 ƒlayer_regularization_losses
≈layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses* 

∆trace_0* 

«trace_0* 

Ь0
Э1*

Ь0
Э1*
* 
Ю
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses*

Ќtrace_0* 

ќtrace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
¶0
І1
®2
©3*

¶0
І1*
* 
Ю
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
Я	variables
†trainable_variables
°regularization_losses
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses*

‘trace_0
’trace_1* 

÷trace_0
„trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
™	variables
Ђtrainable_variables
ђregularization_losses
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses* 

Ёtrace_0* 

ёtrace_0* 

ґ0
Ј1*

ґ0
Ј1*
* 
Ю
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
∞	variables
±trainable_variables
≤regularization_losses
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses*

дtrace_0* 

еtrace_0* 
`Z
VARIABLE_VALUEconv2d_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
ј0
Ѕ1
¬2
√3*

ј0
Ѕ1*
* 
Ю
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
є	variables
Їtrainable_variables
їregularization_losses
љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses*

лtrace_0
мtrace_1* 

нtrace_0
оtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
ƒ	variables
≈trainable_variables
∆regularization_losses
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses* 

фtrace_0* 

хtrace_0* 

–0
—1*

–0
—1*
* 
Ю
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
 	variables
Ћtrainable_variables
ћregularization_losses
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses*

ыtrace_0* 

ьtrace_0* 
`Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Џ0
џ1
№2
Ё3*

Џ0
џ1*
* 
Ю
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
”	variables
‘trainable_variables
’regularization_losses
„__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*

Вtrace_0
Гtrace_1* 

Дtrace_0
Еtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
ё	variables
яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses* 

Лtrace_0* 

Мtrace_0* 

к0
л1*

к0
л1*
* 
Ю
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses*

Тtrace_0* 

Уtrace_0* 
`Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
ф0
х1
ц2
ч3*

ф0
х1*
* 
Ю
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses*

Щtrace_0
Ъtrace_1* 

Ыtrace_0
Ьtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses* 

Ґtrace_0* 

£trace_0* 

Д0
Е1*

Д0
Е1*
* 
Ю
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
ю	variables
€trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses*

©trace_0* 

™trace_0* 
`Z
VARIABLE_VALUEconv2d_8/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_8/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
О0
П1
Р2
С3*

О0
П1*
* 
Ю
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses*

∞trace_0
±trace_1* 

≤trace_0
≥trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses* 

єtrace_0* 

Їtrace_0* 

Ю0
Я1*

Ю0
Я1*
* 
Ю
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses*

јtrace_0* 

Ѕtrace_0* 
`Z
VARIABLE_VALUEconv2d_9/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_9/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
®0
©1
™2
Ђ3*

®0
©1*
* 
Ю
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
°	variables
Ґtrainable_variables
£regularization_losses
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses*

«trace_0
»trace_1* 

…trace_0
 trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_9/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_9/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_9/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_9/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Ћnon_trainable_variables
ћlayers
Ќmetrics
 ќlayer_regularization_losses
ѕlayer_metrics
ђ	variables
≠trainable_variables
Ѓregularization_losses
∞__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses* 

–trace_0* 

—trace_0* 

Є0
є1*

Є0
є1*
* 
Ю
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
≤	variables
≥trainable_variables
іregularization_losses
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses*

„trace_0* 

Ўtrace_0* 
a[
VARIABLE_VALUEconv2d_10/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_10/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
¬0
√1
ƒ2
≈3*

¬0
√1*
* 
Ю
ўnon_trainable_variables
Џlayers
џmetrics
 №layer_regularization_losses
Ёlayer_metrics
ї	variables
Љtrainable_variables
љregularization_losses
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses*

ёtrace_0
яtrace_1* 

аtrace_0
бtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_10/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_10/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_10/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE&batch_normalization_10/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
∆	variables
«trainable_variables
»regularization_losses
 __call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses* 

зtrace_0* 

иtrace_0* 

“0
”1*

“0
”1*
* 
Ю
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
ћ	variables
Ќtrainable_variables
ќregularization_losses
–__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses*

оtrace_0* 

пtrace_0* 
a[
VARIABLE_VALUEconv2d_11/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_11/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
’	variables
÷trainable_variables
„regularization_losses
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses* 

хtrace_0* 

цtrace_0* 
Ї
@0
A1
Z2
[3
t4
u5
О6
П7
®8
©9
¬10
√11
№12
Ё13
ц14
ч15
Р16
С17
™18
Ђ19
ƒ20
≈21*
Ъ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35*

ч0*
* 
* 
* 
* 
* 
* 
њ
е0
ш1
щ2
ъ3
ы4
ь5
э6
ю7
€8
А9
Б10
В11
Г12
Д13
Е14
Ж15
З16
И17
Й18
К19
Л20
М21
Н22
О23
П24
Р25
С26
Т27
У28
Ф29
Х30
Ц31
Ч32
Ш33
Щ34
Ъ35
Ы36
Ь37
Э38
Ю39
Я40
†41
°42
Ґ43
£44
§45
•46
¶47
І48
®49
©50
™51
Ђ52
ђ53
≠54
Ѓ55
ѓ56
∞57
±58
≤59
≥60
і61
µ62
ґ63
Ј64
Є65
є66
Ї67
ї68
Љ69
љ70
Њ71
њ72
ј73
Ѕ74
¬75
√76
ƒ77
≈78
∆79
«80
»81
…82
 83
Ћ84
ћ85
Ќ86
ќ87
ѕ88
–89
—90
“91
”92*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ш
ш0
ъ1
ь2
ю3
А4
В5
Д6
Ж7
И8
К9
М10
О11
Р12
Т13
Ф14
Ц15
Ш16
Ъ17
Ь18
Ю19
†20
Ґ21
§22
¶23
®24
™25
ђ26
Ѓ27
∞28
≤29
і30
ґ31
Є32
Ї33
Љ34
Њ35
ј36
¬37
ƒ38
∆39
»40
 41
ћ42
ќ43
–44
“45*
Ш
щ0
ы1
э2
€3
Б4
Г5
Е6
З7
Й8
Л9
Н10
П11
С12
У13
Х14
Ч15
Щ16
Ы17
Э18
Я19
°20
£21
•22
І23
©24
Ђ25
≠26
ѓ27
±28
≥29
µ30
Ј31
є32
ї33
љ34
њ35
Ѕ36
√37
≈38
«39
…40
Ћ41
Ќ42
ѕ43
—44
”45*
™
‘trace_0
’trace_1
÷trace_2
„trace_3
Ўtrace_4
ўtrace_5
Џtrace_6
џtrace_7
№trace_8
Ёtrace_9
ёtrace_10
яtrace_11
аtrace_12
бtrace_13
вtrace_14
гtrace_15
дtrace_16
еtrace_17
жtrace_18
зtrace_19
иtrace_20
йtrace_21
кtrace_22
лtrace_23
мtrace_24
нtrace_25
оtrace_26
пtrace_27
рtrace_28
сtrace_29
тtrace_30
уtrace_31
фtrace_32
хtrace_33
цtrace_34
чtrace_35
шtrace_36
щtrace_37
ъtrace_38
ыtrace_39
ьtrace_40
эtrace_41
юtrace_42
€trace_43
Аtrace_44
Бtrace_45* 
* 
* 
* 
* 
* 
* 
* 
* 

@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Z0
[1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

t0
u1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

О0
П1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

®0
©1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

¬0
√1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

№0
Ё1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ц0
ч1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Р0
С1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

™0
Ђ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ƒ0
≈1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
В	variables
Г	keras_api

Дtotal

Еcount*
_Y
VARIABLE_VALUEAdam/m/conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_3/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_3/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_3/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_3/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_4/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_4/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_4/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_4/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_4/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_4/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_4/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_4/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_5/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_5/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_5/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_5/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_5/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_5/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_5/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_5/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_6/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_6/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_6/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_6/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_6/gamma2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_6/gamma2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_6/beta2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_6/beta2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_7/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_7/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_7/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_7/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_7/gamma2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_7/gamma2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_7/beta2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_7/beta2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_8/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_8/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_8/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_8/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_8/gamma2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_8/gamma2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_8/beta2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_8/beta2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_9/kernel2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_9/kernel2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_9/bias2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_9/bias2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_9/gamma2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_9/gamma2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_9/beta2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_9/beta2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_10/kernel2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_10/kernel2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_10/bias2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_10/bias2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_10/gamma2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_10/gamma2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_10/beta2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_10/beta2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_11/kernel2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_11/kernel2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_11/bias2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_11/bias2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Д0
Е1*

В	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
р&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/bias	iterationlearning_rateAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv2d_2/kernelAdam/v/conv2d_2/kernelAdam/m/conv2d_2/biasAdam/v/conv2d_2/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/bias"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/bias"Adam/m/batch_normalization_4/gamma"Adam/v/batch_normalization_4/gamma!Adam/m/batch_normalization_4/beta!Adam/v/batch_normalization_4/betaAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/bias"Adam/m/batch_normalization_5/gamma"Adam/v/batch_normalization_5/gamma!Adam/m/batch_normalization_5/beta!Adam/v/batch_normalization_5/betaAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias"Adam/m/batch_normalization_6/gamma"Adam/v/batch_normalization_6/gamma!Adam/m/batch_normalization_6/beta!Adam/v/batch_normalization_6/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias"Adam/m/batch_normalization_7/gamma"Adam/v/batch_normalization_7/gamma!Adam/m/batch_normalization_7/beta!Adam/v/batch_normalization_7/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias"Adam/m/batch_normalization_8/gamma"Adam/v/batch_normalization_8/gamma!Adam/m/batch_normalization_8/beta!Adam/v/batch_normalization_8/betaAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/bias"Adam/m/batch_normalization_9/gamma"Adam/v/batch_normalization_9/gamma!Adam/m/batch_normalization_9/beta!Adam/v/batch_normalization_9/betaAdam/m/conv2d_10/kernelAdam/v/conv2d_10/kernelAdam/m/conv2d_10/biasAdam/v/conv2d_10/bias#Adam/m/batch_normalization_10/gamma#Adam/v/batch_normalization_10/gamma"Adam/m/batch_normalization_10/beta"Adam/v/batch_normalization_10/betaAdam/m/conv2d_11/kernelAdam/v/conv2d_11/kernelAdam/m/conv2d_11/biasAdam/v/conv2d_11/biastotalcountConst*і
Tinђ
©2¶*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_21669
л&
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/bias	iterationlearning_rateAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv2d_2/kernelAdam/v/conv2d_2/kernelAdam/m/conv2d_2/biasAdam/v/conv2d_2/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/bias"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/bias"Adam/m/batch_normalization_4/gamma"Adam/v/batch_normalization_4/gamma!Adam/m/batch_normalization_4/beta!Adam/v/batch_normalization_4/betaAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/bias"Adam/m/batch_normalization_5/gamma"Adam/v/batch_normalization_5/gamma!Adam/m/batch_normalization_5/beta!Adam/v/batch_normalization_5/betaAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias"Adam/m/batch_normalization_6/gamma"Adam/v/batch_normalization_6/gamma!Adam/m/batch_normalization_6/beta!Adam/v/batch_normalization_6/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias"Adam/m/batch_normalization_7/gamma"Adam/v/batch_normalization_7/gamma!Adam/m/batch_normalization_7/beta!Adam/v/batch_normalization_7/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias"Adam/m/batch_normalization_8/gamma"Adam/v/batch_normalization_8/gamma!Adam/m/batch_normalization_8/beta!Adam/v/batch_normalization_8/betaAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/bias"Adam/m/batch_normalization_9/gamma"Adam/v/batch_normalization_9/gamma!Adam/m/batch_normalization_9/beta!Adam/v/batch_normalization_9/betaAdam/m/conv2d_10/kernelAdam/v/conv2d_10/kernelAdam/m/conv2d_10/biasAdam/v/conv2d_10/bias#Adam/m/batch_normalization_10/gamma#Adam/v/batch_normalization_10/gamma"Adam/m/batch_normalization_10/beta"Adam/v/batch_normalization_10/betaAdam/m/conv2d_11/kernelAdam/v/conv2d_11/kernelAdam/m/conv2d_11/biasAdam/v/conv2d_11/biastotalcount*≥
TinЂ
®2•*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_22170кє$
З
љ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17611

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
±
ь
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19832

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
а
Ю
)__inference_conv2d_10_layer_call_fn_20550

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_18587Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20544:%!

_user_specified_name20546
±
ь
C__inference_conv2d_9_layer_call_and_return_conditional_losses_20469

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19581
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
Ы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20258

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Р3
»
#__inference_signature_wrapper_19401
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@@

unknown_48:@

unknown_49:@

unknown_50:@

unknown_51:@

unknown_52:@$

unknown_53:@@

unknown_54:@

unknown_55:@

unknown_56:@

unknown_57:@

unknown_58:@$

unknown_59:@@

unknown_60:@

unknown_61:@

unknown_62:@

unknown_63:@

unknown_64:@$

unknown_65:@

unknown_66:
identityИҐStatefulPartitionedCallж	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_17593Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapesЄ
µ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:%!

_user_specified_name19263:%!

_user_specified_name19265:%!

_user_specified_name19267:%!

_user_specified_name19269:%!

_user_specified_name19271:%!

_user_specified_name19273:%!

_user_specified_name19275:%!

_user_specified_name19277:%	!

_user_specified_name19279:%
!

_user_specified_name19281:%!

_user_specified_name19283:%!

_user_specified_name19285:%!

_user_specified_name19287:%!

_user_specified_name19289:%!

_user_specified_name19291:%!

_user_specified_name19293:%!

_user_specified_name19295:%!

_user_specified_name19297:%!

_user_specified_name19299:%!

_user_specified_name19301:%!

_user_specified_name19303:%!

_user_specified_name19305:%!

_user_specified_name19307:%!

_user_specified_name19309:%!

_user_specified_name19311:%!

_user_specified_name19313:%!

_user_specified_name19315:%!

_user_specified_name19317:%!

_user_specified_name19319:%!

_user_specified_name19321:%!

_user_specified_name19323:% !

_user_specified_name19325:%!!

_user_specified_name19327:%"!

_user_specified_name19329:%#!

_user_specified_name19331:%$!

_user_specified_name19333:%%!

_user_specified_name19335:%&!

_user_specified_name19337:%'!

_user_specified_name19339:%(!

_user_specified_name19341:%)!

_user_specified_name19343:%*!

_user_specified_name19345:%+!

_user_specified_name19347:%,!

_user_specified_name19349:%-!

_user_specified_name19351:%.!

_user_specified_name19353:%/!

_user_specified_name19355:%0!

_user_specified_name19357:%1!

_user_specified_name19359:%2!

_user_specified_name19361:%3!

_user_specified_name19363:%4!

_user_specified_name19365:%5!

_user_specified_name19367:%6!

_user_specified_name19369:%7!

_user_specified_name19371:%8!

_user_specified_name19373:%9!

_user_specified_name19375:%:!

_user_specified_name19377:%;!

_user_specified_name19379:%<!

_user_specified_name19381:%=!

_user_specified_name19383:%>!

_user_specified_name19385:%?!

_user_specified_name19387:%@!

_user_specified_name19389:%A!

_user_specified_name19391:%B!

_user_specified_name19393:%C!

_user_specified_name19395:%D!

_user_specified_name19397
а
Ю
)__inference_conv2d_11_layer_call_fn_20641

inputs!
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_18617Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20635:%!

_user_specified_name20637
ё
Э
(__inference_conv2d_3_layer_call_fn_19913

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_18377Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19907:%!

_user_specified_name19909
ѕ
Ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17691

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17877

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
±
ь
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19741

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѓ
ъ
A__inference_conv2d_layer_call_and_return_conditional_losses_19650

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
^
B__inference_re_lu_7_layer_call_and_return_conditional_losses_18516

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
К

–
5__inference_batch_normalization_2_layer_call_fn_19858

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_17753Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19848:%!

_user_specified_name19850:%!

_user_specified_name19852:%!

_user_specified_name19854
ё
Э
(__inference_conv2d_4_layer_call_fn_20004

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_18407Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19998:%!

_user_specified_name20000
Ѓ
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_19813

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
±
ь
C__inference_conv2d_9_layer_call_and_return_conditional_losses_18557

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Д
C
'__inference_re_lu_6_layer_call_fn_20263

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_18486z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Д
C
'__inference_re_lu_4_layer_call_fn_20081

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_18426z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ѓ
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_18366

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_19571
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
Ы
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20531

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_18336

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
К

–
5__inference_batch_normalization_1_layer_call_fn_19767

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17691Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19757:%!

_user_specified_name19759:%!

_user_specified_name19761:%!

_user_specified_name19763
Ѓ
^
B__inference_re_lu_6_layer_call_and_return_conditional_losses_20268

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ѕ
Ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20349

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѓ
ъ
A__inference_conv2d_layer_call_and_return_conditional_losses_18287

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Й
њ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20058

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19591
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19421
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
h
>__inference_add_layer_call_and_return_conditional_losses_18628

inputs
inputs_1
identityj
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
М

—
6__inference_batch_normalization_10_layer_call_fn_20586

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_18249Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20576:%!

_user_specified_name20578:%!

_user_specified_name20580:%!

_user_specified_name20582
±
ь
C__inference_conv2d_4_layer_call_and_return_conditional_losses_18407

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
–
Ь
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20622

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Џ
Ы
&__inference_conv2d_layer_call_fn_19640

inputs!
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_18287Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:%!

_user_specified_name19634:%!

_user_specified_name19636
Й
њ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18045

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19481
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
V
"__inference__update_step_xla_19546
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
И

–
5__inference_batch_normalization_3_layer_call_fn_19936

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_17797Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19926:%!

_user_specified_name19928:%!

_user_specified_name19930:%!

_user_specified_name19932
Й
њ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_17735

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19501
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19531
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
К
ј
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_18231

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
V
"__inference__update_step_xla_19506
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19541
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
њ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_17797

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ё
Э
(__inference_conv2d_1_layer_call_fn_19731

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18317Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19725:%!

_user_specified_name19727
ё
Э
(__inference_conv2d_5_layer_call_fn_20095

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_18437Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20089:%!

_user_specified_name20091
Ж

ќ
3__inference_batch_normalization_layer_call_fn_19676

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17629Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19666:%!

_user_specified_name19668:%!

_user_specified_name19670:%!

_user_specified_name19672
И

–
5__inference_batch_normalization_8_layer_call_fn_20391

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_18107Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20381:%!

_user_specified_name20383:%!

_user_specified_name20385:%!

_user_specified_name20387
Й
њ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18169

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20167

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ўг
ƒA
 __inference__wrapped_model_17593
input_1E
+model_conv2d_conv2d_readvariableop_resource:@:
,model_conv2d_biasadd_readvariableop_resource:@?
1model_batch_normalization_readvariableop_resource:@A
3model_batch_normalization_readvariableop_1_resource:@P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:@R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_1_conv2d_readvariableop_resource:@@<
.model_conv2d_1_biasadd_readvariableop_resource:@A
3model_batch_normalization_1_readvariableop_resource:@C
5model_batch_normalization_1_readvariableop_1_resource:@R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_2_conv2d_readvariableop_resource:@@<
.model_conv2d_2_biasadd_readvariableop_resource:@A
3model_batch_normalization_2_readvariableop_resource:@C
5model_batch_normalization_2_readvariableop_1_resource:@R
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_3_conv2d_readvariableop_resource:@@<
.model_conv2d_3_biasadd_readvariableop_resource:@A
3model_batch_normalization_3_readvariableop_resource:@C
5model_batch_normalization_3_readvariableop_1_resource:@R
Dmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_4_conv2d_readvariableop_resource:@@<
.model_conv2d_4_biasadd_readvariableop_resource:@A
3model_batch_normalization_4_readvariableop_resource:@C
5model_batch_normalization_4_readvariableop_1_resource:@R
Dmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_5_conv2d_readvariableop_resource:@@<
.model_conv2d_5_biasadd_readvariableop_resource:@A
3model_batch_normalization_5_readvariableop_resource:@C
5model_batch_normalization_5_readvariableop_1_resource:@R
Dmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_6_conv2d_readvariableop_resource:@@<
.model_conv2d_6_biasadd_readvariableop_resource:@A
3model_batch_normalization_6_readvariableop_resource:@C
5model_batch_normalization_6_readvariableop_1_resource:@R
Dmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_7_conv2d_readvariableop_resource:@@<
.model_conv2d_7_biasadd_readvariableop_resource:@A
3model_batch_normalization_7_readvariableop_resource:@C
5model_batch_normalization_7_readvariableop_1_resource:@R
Dmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_8_conv2d_readvariableop_resource:@@<
.model_conv2d_8_biasadd_readvariableop_resource:@A
3model_batch_normalization_8_readvariableop_resource:@C
5model_batch_normalization_8_readvariableop_1_resource:@R
Dmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_9_conv2d_readvariableop_resource:@@<
.model_conv2d_9_biasadd_readvariableop_resource:@A
3model_batch_normalization_9_readvariableop_resource:@C
5model_batch_normalization_9_readvariableop_1_resource:@R
Dmodel_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@H
.model_conv2d_10_conv2d_readvariableop_resource:@@=
/model_conv2d_10_biasadd_readvariableop_resource:@B
4model_batch_normalization_10_readvariableop_resource:@D
6model_batch_normalization_10_readvariableop_1_resource:@S
Emodel_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@U
Gmodel_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@H
.model_conv2d_11_conv2d_readvariableop_resource:@=
/model_conv2d_11_biasadd_readvariableop_resource:
identityИҐ9model/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ(model/batch_normalization/ReadVariableOpҐ*model/batch_normalization/ReadVariableOp_1Ґ;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_1/ReadVariableOpҐ,model/batch_normalization_1/ReadVariableOp_1Ґ<model/batch_normalization_10/FusedBatchNormV3/ReadVariableOpҐ>model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ґ+model/batch_normalization_10/ReadVariableOpҐ-model/batch_normalization_10/ReadVariableOp_1Ґ;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_2/ReadVariableOpҐ,model/batch_normalization_2/ReadVariableOp_1Ґ;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_3/ReadVariableOpҐ,model/batch_normalization_3/ReadVariableOp_1Ґ;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_4/ReadVariableOpҐ,model/batch_normalization_4/ReadVariableOp_1Ґ;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_5/ReadVariableOpҐ,model/batch_normalization_5/ReadVariableOp_1Ґ;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_6/ReadVariableOpҐ,model/batch_normalization_6/ReadVariableOp_1Ґ;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_7/ReadVariableOpҐ,model/batch_normalization_7/ReadVariableOp_1Ґ;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_8/ReadVariableOpҐ,model/batch_normalization_8/ReadVariableOp_1Ґ;model/batch_normalization_9/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_9/ReadVariableOpҐ,model/batch_normalization_9/ReadVariableOp_1Ґ#model/conv2d/BiasAdd/ReadVariableOpҐ"model/conv2d/Conv2D/ReadVariableOpҐ%model/conv2d_1/BiasAdd/ReadVariableOpҐ$model/conv2d_1/Conv2D/ReadVariableOpҐ&model/conv2d_10/BiasAdd/ReadVariableOpҐ%model/conv2d_10/Conv2D/ReadVariableOpҐ&model/conv2d_11/BiasAdd/ReadVariableOpҐ%model/conv2d_11/Conv2D/ReadVariableOpҐ%model/conv2d_2/BiasAdd/ReadVariableOpҐ$model/conv2d_2/Conv2D/ReadVariableOpҐ%model/conv2d_3/BiasAdd/ReadVariableOpҐ$model/conv2d_3/Conv2D/ReadVariableOpҐ%model/conv2d_4/BiasAdd/ReadVariableOpҐ$model/conv2d_4/Conv2D/ReadVariableOpҐ%model/conv2d_5/BiasAdd/ReadVariableOpҐ$model/conv2d_5/Conv2D/ReadVariableOpҐ%model/conv2d_6/BiasAdd/ReadVariableOpҐ$model/conv2d_6/Conv2D/ReadVariableOpҐ%model/conv2d_7/BiasAdd/ReadVariableOpҐ$model/conv2d_7/Conv2D/ReadVariableOpҐ%model/conv2d_8/BiasAdd/ReadVariableOpҐ$model/conv2d_8/Conv2D/ReadVariableOpҐ%model/conv2d_9/BiasAdd/ReadVariableOpҐ$model/conv2d_9/Conv2D/ReadVariableOpЦ
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0∆
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
М
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ґ
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ц
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0Є
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0б
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ф
model/re_lu/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0б
model/conv2d_1/Conv2DConv2Dmodel/re_lu/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d_1/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_2/Conv2DConv2D model/re_lu_1/Relu:activations:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3model/conv2d_2/BiasAdd:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_2/ReluRelu0model/batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_3/Conv2DConv2D model/re_lu_2/Relu:activations:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_3/ReadVariableOpReadVariableOp3model_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOp5model_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3model/conv2d_3/BiasAdd:output:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_3/ReluRelu0model/batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_4/Conv2DConv2D model/re_lu_3/Relu:activations:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_4/ReadVariableOpReadVariableOp3model_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_4/ReadVariableOp_1ReadVariableOp5model_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3model/conv2d_4/BiasAdd:output:02model/batch_normalization_4/ReadVariableOp:value:04model/batch_normalization_4/ReadVariableOp_1:value:0Cmodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_4/ReluRelu0model/batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_5/Conv2DConv2D model/re_lu_4/Relu:activations:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_5/ReadVariableOpReadVariableOp3model_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_5/ReadVariableOp_1ReadVariableOp5model_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3model/conv2d_5/BiasAdd:output:02model/batch_normalization_5/ReadVariableOp:value:04model/batch_normalization_5/ReadVariableOp_1:value:0Cmodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_5/ReluRelu0model/batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_6/Conv2DConv2D model/re_lu_5/Relu:activations:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_6/ReadVariableOpReadVariableOp3model_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_6/ReadVariableOp_1ReadVariableOp5model_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3model/conv2d_6/BiasAdd:output:02model/batch_normalization_6/ReadVariableOp:value:04model/batch_normalization_6/ReadVariableOp_1:value:0Cmodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_6/ReluRelu0model/batch_normalization_6/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_7/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_7/Conv2DConv2D model/re_lu_6/Relu:activations:0,model/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_7/BiasAddBiasAddmodel/conv2d_7/Conv2D:output:0-model/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_7/ReadVariableOpReadVariableOp3model_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_7/ReadVariableOp_1ReadVariableOp5model_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3model/conv2d_7/BiasAdd:output:02model/batch_normalization_7/ReadVariableOp:value:04model/batch_normalization_7/ReadVariableOp_1:value:0Cmodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_7/ReluRelu0model/batch_normalization_7/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_8/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_8/Conv2DConv2D model/re_lu_7/Relu:activations:0,model/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_8/BiasAddBiasAddmodel/conv2d_8/Conv2D:output:0-model/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_8/ReadVariableOpReadVariableOp3model_batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_8/ReadVariableOp_1ReadVariableOp5model_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3model/conv2d_8/BiasAdd:output:02model/batch_normalization_8/ReadVariableOp:value:04model/batch_normalization_8/ReadVariableOp_1:value:0Cmodel/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_8/ReluRelu0model/batch_normalization_8/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
$model/conv2d_9/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
model/conv2d_9/Conv2DConv2D model/re_lu_8/Relu:activations:0,model/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Р
%model/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/conv2d_9/BiasAddBiasAddmodel/conv2d_9/Conv2D:output:0-model/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ъ
*model/batch_normalization_9/ReadVariableOpReadVariableOp3model_batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
,model/batch_normalization_9/ReadVariableOp_1ReadVariableOp5model_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ј
=model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
,model/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3model/conv2d_9/BiasAdd:output:02model/batch_normalization_9/ReadVariableOp:value:04model/batch_normalization_9/ReadVariableOp_1:value:0Cmodel/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ш
model/re_lu_9/ReluRelu0model/batch_normalization_9/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
%model/conv2d_10/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0е
model/conv2d_10/Conv2DConv2D model/re_lu_9/Relu:activations:0-model/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Т
&model/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
model/conv2d_10/BiasAddBiasAddmodel/conv2d_10/Conv2D:output:0.model/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
+model/batch_normalization_10/ReadVariableOpReadVariableOp4model_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0†
-model/batch_normalization_10/ReadVariableOp_1ReadVariableOp6model_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<model/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0у
-model/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3 model/conv2d_10/BiasAdd:output:03model/batch_normalization_10/ReadVariableOp:value:05model/batch_normalization_10/ReadVariableOp_1:value:0Dmodel/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ъ
model/re_lu_10/ReluRelu1model/batch_normalization_10/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
%model/conv2d_11/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ж
model/conv2d_11/Conv2DConv2D!model/re_lu_10/Relu:activations:0-model/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
Т
&model/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0њ
model/conv2d_11/BiasAddBiasAddmodel/conv2d_11/Conv2D:output:0.model/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
model/add/addAddV2input_1 model/conv2d_11/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€z
IdentityIdentitymodel/add/add:z:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€∞
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1=^model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_10/ReadVariableOp.^model/batch_normalization_10/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1<^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_3/ReadVariableOp-^model/batch_normalization_3/ReadVariableOp_1<^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_4/ReadVariableOp-^model/batch_normalization_4/ReadVariableOp_1<^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_5/ReadVariableOp-^model/batch_normalization_5/ReadVariableOp_1<^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_6/ReadVariableOp-^model/batch_normalization_6/ReadVariableOp_1<^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_7/ReadVariableOp-^model/batch_normalization_7/ReadVariableOp_1<^model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_8/ReadVariableOp-^model/batch_normalization_8/ReadVariableOp_1<^model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_9/ReadVariableOp-^model/batch_normalization_9/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_10/BiasAdd/ReadVariableOp&^model/conv2d_10/Conv2D/ReadVariableOp'^model/conv2d_11/BiasAdd/ReadVariableOp&^model/conv2d_11/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp&^model/conv2d_7/BiasAdd/ReadVariableOp%^model/conv2d_7/Conv2D/ReadVariableOp&^model/conv2d_8/BiasAdd/ReadVariableOp%^model/conv2d_8/Conv2D/ReadVariableOp&^model/conv2d_9/BiasAdd/ReadVariableOp%^model/conv2d_9/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapesЄ
µ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12|
<model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2А
>model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_10/ReadVariableOp+model/batch_normalization_10/ReadVariableOp2^
-model/batch_normalization_10/ReadVariableOp_1-model/batch_normalization_10/ReadVariableOp_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12z
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_3/ReadVariableOp*model/batch_normalization_3/ReadVariableOp2\
,model/batch_normalization_3/ReadVariableOp_1,model/batch_normalization_3/ReadVariableOp_12z
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_4/ReadVariableOp*model/batch_normalization_4/ReadVariableOp2\
,model/batch_normalization_4/ReadVariableOp_1,model/batch_normalization_4/ReadVariableOp_12z
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_5/ReadVariableOp*model/batch_normalization_5/ReadVariableOp2\
,model/batch_normalization_5/ReadVariableOp_1,model/batch_normalization_5/ReadVariableOp_12z
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_6/ReadVariableOp*model/batch_normalization_6/ReadVariableOp2\
,model/batch_normalization_6/ReadVariableOp_1,model/batch_normalization_6/ReadVariableOp_12z
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_7/ReadVariableOp*model/batch_normalization_7/ReadVariableOp2\
,model/batch_normalization_7/ReadVariableOp_1,model/batch_normalization_7/ReadVariableOp_12z
;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_8/ReadVariableOp*model/batch_normalization_8/ReadVariableOp2\
,model/batch_normalization_8/ReadVariableOp_1,model/batch_normalization_8/ReadVariableOp_12z
;model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_9/ReadVariableOp*model/batch_normalization_9/ReadVariableOp2\
,model/batch_normalization_9/ReadVariableOp_1,model/batch_normalization_9/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_10/BiasAdd/ReadVariableOp&model/conv2d_10/BiasAdd/ReadVariableOp2N
%model/conv2d_10/Conv2D/ReadVariableOp%model/conv2d_10/Conv2D/ReadVariableOp2P
&model/conv2d_11/BiasAdd/ReadVariableOp&model/conv2d_11/BiasAdd/ReadVariableOp2N
%model/conv2d_11/Conv2D/ReadVariableOp%model/conv2d_11/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2N
%model/conv2d_7/BiasAdd/ReadVariableOp%model/conv2d_7/BiasAdd/ReadVariableOp2L
$model/conv2d_7/Conv2D/ReadVariableOp$model/conv2d_7/Conv2D/ReadVariableOp2N
%model/conv2d_8/BiasAdd/ReadVariableOp%model/conv2d_8/BiasAdd/ReadVariableOp2L
$model/conv2d_8/Conv2D/ReadVariableOp$model/conv2d_8/Conv2D/ReadVariableOp2N
%model/conv2d_9/BiasAdd/ReadVariableOp%model/conv2d_9/BiasAdd/ReadVariableOp2L
$model/conv2d_9/Conv2D/ReadVariableOp$model/conv2d_9/Conv2D/ReadVariableOp:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(7$
"
_user_specified_name
resource:(8$
"
_user_specified_name
resource:(9$
"
_user_specified_name
resource:(:$
"
_user_specified_name
resource:(;$
"
_user_specified_name
resource:(<$
"
_user_specified_name
resource:(=$
"
_user_specified_name
resource:(>$
"
_user_specified_name
resource:(?$
"
_user_specified_name
resource:(@$
"
_user_specified_name
resource:(A$
"
_user_specified_name
resource:(B$
"
_user_specified_name
resource:(C$
"
_user_specified_name
resource:(D$
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19496
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19561
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
Ы
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18187

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_17815

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19616
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
З
љ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19694

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
^
B__inference_re_lu_9_layer_call_and_return_conditional_losses_18576

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Д
C
'__inference_re_lu_5_layer_call_fn_20172

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_18456z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
К
ј
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20604

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
К

–
5__inference_batch_normalization_8_layer_call_fn_20404

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_18125Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20394:%!

_user_specified_name20396:%!

_user_specified_name20398:%!

_user_specified_name20400
Ќ
Щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17629

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
^
B__inference_re_lu_7_layer_call_and_return_conditional_losses_20359

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ё
Э
(__inference_conv2d_8_layer_call_fn_20368

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_18527Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20362:%!

_user_specified_name20364
ѕ
V
"__inference__update_step_xla_19606
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
И

–
5__inference_batch_normalization_7_layer_call_fn_20300

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18045Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20290:%!

_user_specified_name20292:%!

_user_specified_name20294:%!

_user_specified_name20296
И

–
5__inference_batch_normalization_5_layer_call_fn_20118

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_17921Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20108:%!

_user_specified_name20110:%!

_user_specified_name20112:%!

_user_specified_name20114
Ђ
J
"__inference__update_step_xla_19596
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19536
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
А
A
%__inference_re_lu_layer_call_fn_19717

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_18306z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Й
њ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19785

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
±
ь
C__inference_conv2d_7_layer_call_and_return_conditional_losses_18497

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_17753

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19803

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19441
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
±
ь
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18317

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
^
B__inference_re_lu_6_layer_call_and_return_conditional_losses_18486

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
К

–
5__inference_batch_normalization_3_layer_call_fn_19949

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_17815Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19939:%!

_user_specified_name19941:%!

_user_specified_name19943:%!

_user_specified_name19945
Ѓ
^
B__inference_re_lu_8_layer_call_and_return_conditional_losses_18546

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_19436
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
≤
э
D__inference_conv2d_11_layer_call_and_return_conditional_losses_18617

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
V
"__inference__update_step_xla_19586
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19551
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ѓ
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_20177

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_19461
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
њ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20422

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_17939

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
V
"__inference__update_step_xla_19446
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
њ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20240

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Й
њ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19967

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Й
њ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17673

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
^
B__inference_re_lu_9_layer_call_and_return_conditional_losses_20541

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
±
ь
C__inference_conv2d_3_layer_call_and_return_conditional_losses_18377

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19556
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
≤
э
D__inference_conv2d_10_layer_call_and_return_conditional_losses_20560

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
±
ь
C__inference_conv2d_5_layer_call_and_return_conditional_losses_18437

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
К

–
5__inference_batch_normalization_4_layer_call_fn_20040

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17877Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20030:%!

_user_specified_name20032:%!

_user_specified_name20034:%!

_user_specified_name20036
Ђ
J
"__inference__update_step_xla_19471
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
±
ь
C__inference_conv2d_6_layer_call_and_return_conditional_losses_20196

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19601
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ё
Э
(__inference_conv2d_7_layer_call_fn_20277

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_18497Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20271:%!

_user_specified_name20273
Ђ
J
"__inference__update_step_xla_19416
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ѓ
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_18456

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ѓ
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_18396

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И

–
5__inference_batch_normalization_2_layer_call_fn_19845

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_17735Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19835:%!

_user_specified_name19837:%!

_user_specified_name19839:%!

_user_specified_name19841
ѕ
Ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18063

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
К

–
5__inference_batch_normalization_6_layer_call_fn_20222

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_18001Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20212:%!

_user_specified_name20214:%!

_user_specified_name20216:%!

_user_specified_name20218
Ђ
J
"__inference__update_step_xla_19511
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
њ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20149

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ђ
\
@__inference_re_lu_layer_call_and_return_conditional_losses_18306

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И

–
5__inference_batch_normalization_4_layer_call_fn_20027

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17859Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20017:%!

_user_specified_name20019:%!

_user_specified_name20021:%!

_user_specified_name20023
ѓ
_
C__inference_re_lu_10_layer_call_and_return_conditional_losses_20632

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_19521
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Д
C
'__inference_re_lu_9_layer_call_fn_20536

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_18576z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
К

–
5__inference_batch_normalization_7_layer_call_fn_20313

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18063Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20303:%!

_user_specified_name20305:%!

_user_specified_name20307:%!

_user_specified_name20309
±
ь
C__inference_conv2d_6_layer_call_and_return_conditional_losses_18467

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
V
"__inference__update_step_xla_19426
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
њ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_17921

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
±
ь
C__inference_conv2d_4_layer_call_and_return_conditional_losses_20014

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Д
C
'__inference_re_lu_1_layer_call_fn_19808

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_18336z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ѕ
V
"__inference__update_step_xla_19406
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:P L
&
_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
±
ь
C__inference_conv2d_2_layer_call_and_return_conditional_losses_18347

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Й
њ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19876

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
≤
э
D__inference_conv2d_10_layer_call_and_return_conditional_losses_18587

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19985

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ќ
Щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19712

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Й
њ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_18107

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
К

–
5__inference_batch_normalization_5_layer_call_fn_20131

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_17939Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20121:%!

_user_specified_name20123:%!

_user_specified_name20125:%!

_user_specified_name20127
Ь3
 
%__inference_model_layer_call_fn_18947
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@@

unknown_48:@

unknown_49:@

unknown_50:@

unknown_51:@

unknown_52:@$

unknown_53:@@

unknown_54:@

unknown_55:@

unknown_56:@

unknown_57:@

unknown_58:@$

unknown_59:@@

unknown_60:@

unknown_61:@

unknown_62:@

unknown_63:@

unknown_64:@$

unknown_65:@

unknown_66:
identityИҐStatefulPartitionedCallр	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*P
_read_only_resource_inputs2
0.	
 !"%&'(+,-.1234789:=>?@CD*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_18631Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapesЄ
µ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:%!

_user_specified_name18809:%!

_user_specified_name18811:%!

_user_specified_name18813:%!

_user_specified_name18815:%!

_user_specified_name18817:%!

_user_specified_name18819:%!

_user_specified_name18821:%!

_user_specified_name18823:%	!

_user_specified_name18825:%
!

_user_specified_name18827:%!

_user_specified_name18829:%!

_user_specified_name18831:%!

_user_specified_name18833:%!

_user_specified_name18835:%!

_user_specified_name18837:%!

_user_specified_name18839:%!

_user_specified_name18841:%!

_user_specified_name18843:%!

_user_specified_name18845:%!

_user_specified_name18847:%!

_user_specified_name18849:%!

_user_specified_name18851:%!

_user_specified_name18853:%!

_user_specified_name18855:%!

_user_specified_name18857:%!

_user_specified_name18859:%!

_user_specified_name18861:%!

_user_specified_name18863:%!

_user_specified_name18865:%!

_user_specified_name18867:%!

_user_specified_name18869:% !

_user_specified_name18871:%!!

_user_specified_name18873:%"!

_user_specified_name18875:%#!

_user_specified_name18877:%$!

_user_specified_name18879:%%!

_user_specified_name18881:%&!

_user_specified_name18883:%'!

_user_specified_name18885:%(!

_user_specified_name18887:%)!

_user_specified_name18889:%*!

_user_specified_name18891:%+!

_user_specified_name18893:%,!

_user_specified_name18895:%-!

_user_specified_name18897:%.!

_user_specified_name18899:%/!

_user_specified_name18901:%0!

_user_specified_name18903:%1!

_user_specified_name18905:%2!

_user_specified_name18907:%3!

_user_specified_name18909:%4!

_user_specified_name18911:%5!

_user_specified_name18913:%6!

_user_specified_name18915:%7!

_user_specified_name18917:%8!

_user_specified_name18919:%9!

_user_specified_name18921:%:!

_user_specified_name18923:%;!

_user_specified_name18925:%<!

_user_specified_name18927:%=!

_user_specified_name18929:%>!

_user_specified_name18931:%?!

_user_specified_name18933:%@!

_user_specified_name18935:%A!

_user_specified_name18937:%B!

_user_specified_name18939:%C!

_user_specified_name18941:%D!

_user_specified_name18943
≤
э
D__inference_conv2d_11_layer_call_and_return_conditional_losses_20651

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19621
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19431
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
“«
”
@__inference_model_layer_call_and_return_conditional_losses_18806
input_1&
conv2d_18634:@
conv2d_18636:@'
batch_normalization_18639:@'
batch_normalization_18641:@'
batch_normalization_18643:@'
batch_normalization_18645:@(
conv2d_1_18649:@@
conv2d_1_18651:@)
batch_normalization_1_18654:@)
batch_normalization_1_18656:@)
batch_normalization_1_18658:@)
batch_normalization_1_18660:@(
conv2d_2_18664:@@
conv2d_2_18666:@)
batch_normalization_2_18669:@)
batch_normalization_2_18671:@)
batch_normalization_2_18673:@)
batch_normalization_2_18675:@(
conv2d_3_18679:@@
conv2d_3_18681:@)
batch_normalization_3_18684:@)
batch_normalization_3_18686:@)
batch_normalization_3_18688:@)
batch_normalization_3_18690:@(
conv2d_4_18694:@@
conv2d_4_18696:@)
batch_normalization_4_18699:@)
batch_normalization_4_18701:@)
batch_normalization_4_18703:@)
batch_normalization_4_18705:@(
conv2d_5_18709:@@
conv2d_5_18711:@)
batch_normalization_5_18714:@)
batch_normalization_5_18716:@)
batch_normalization_5_18718:@)
batch_normalization_5_18720:@(
conv2d_6_18724:@@
conv2d_6_18726:@)
batch_normalization_6_18729:@)
batch_normalization_6_18731:@)
batch_normalization_6_18733:@)
batch_normalization_6_18735:@(
conv2d_7_18739:@@
conv2d_7_18741:@)
batch_normalization_7_18744:@)
batch_normalization_7_18746:@)
batch_normalization_7_18748:@)
batch_normalization_7_18750:@(
conv2d_8_18754:@@
conv2d_8_18756:@)
batch_normalization_8_18759:@)
batch_normalization_8_18761:@)
batch_normalization_8_18763:@)
batch_normalization_8_18765:@(
conv2d_9_18769:@@
conv2d_9_18771:@)
batch_normalization_9_18774:@)
batch_normalization_9_18776:@)
batch_normalization_9_18778:@)
batch_normalization_9_18780:@)
conv2d_10_18784:@@
conv2d_10_18786:@*
batch_normalization_10_18789:@*
batch_normalization_10_18791:@*
batch_normalization_10_18793:@*
batch_normalization_10_18795:@)
conv2d_11_18799:@
conv2d_11_18801:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ.batch_normalization_10/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ-batch_normalization_8/StatefulPartitionedCallҐ-batch_normalization_9/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ!conv2d_10/StatefulPartitionedCallҐ!conv2d_11/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐ conv2d_9/StatefulPartitionedCallГ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_18634conv2d_18636*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_18287С
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_18639batch_normalization_18641batch_normalization_18643batch_normalization_18645*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17629ь
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_18306Ґ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_18649conv2d_1_18651*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18317Я
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_18654batch_normalization_1_18656batch_normalization_1_18658batch_normalization_1_18660*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17691В
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_18336§
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_18664conv2d_2_18666*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_18347Я
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_18669batch_normalization_2_18671batch_normalization_2_18673batch_normalization_2_18675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_17753В
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_18366§
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_18679conv2d_3_18681*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_18377Я
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_18684batch_normalization_3_18686batch_normalization_3_18688batch_normalization_3_18690*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_17815В
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_18396§
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_18694conv2d_4_18696*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_18407Я
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_18699batch_normalization_4_18701batch_normalization_4_18703batch_normalization_4_18705*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17877В
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_18426§
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_18709conv2d_5_18711*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_18437Я
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_18714batch_normalization_5_18716batch_normalization_5_18718batch_normalization_5_18720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_17939В
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_18456§
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_6_18724conv2d_6_18726*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_18467Я
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_18729batch_normalization_6_18731batch_normalization_6_18733batch_normalization_6_18735*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_18001В
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_18486§
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0conv2d_7_18739conv2d_7_18741*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_18497Я
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_18744batch_normalization_7_18746batch_normalization_7_18748batch_normalization_7_18750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18063В
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_18516§
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv2d_8_18754conv2d_8_18756*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_18527Я
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_18759batch_normalization_8_18761batch_normalization_8_18763batch_normalization_8_18765*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_18125В
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_18546§
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv2d_9_18769conv2d_9_18771*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_18557Я
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_18774batch_normalization_9_18776batch_normalization_9_18778batch_normalization_9_18780*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18187В
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_18576®
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0conv2d_10_18784conv2d_10_18786*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_18587¶
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_18789batch_normalization_10_18791batch_normalization_10_18793batch_normalization_10_18795*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_18249Е
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_18606©
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0conv2d_11_18799conv2d_11_18801*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_18617ш
add/PartitionedCallPartitionedCallinput_1*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_18628Е
IdentityIdentityadd/PartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€’
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapesЄ
µ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:%!

_user_specified_name18634:%!

_user_specified_name18636:%!

_user_specified_name18639:%!

_user_specified_name18641:%!

_user_specified_name18643:%!

_user_specified_name18645:%!

_user_specified_name18649:%!

_user_specified_name18651:%	!

_user_specified_name18654:%
!

_user_specified_name18656:%!

_user_specified_name18658:%!

_user_specified_name18660:%!

_user_specified_name18664:%!

_user_specified_name18666:%!

_user_specified_name18669:%!

_user_specified_name18671:%!

_user_specified_name18673:%!

_user_specified_name18675:%!

_user_specified_name18679:%!

_user_specified_name18681:%!

_user_specified_name18684:%!

_user_specified_name18686:%!

_user_specified_name18688:%!

_user_specified_name18690:%!

_user_specified_name18694:%!

_user_specified_name18696:%!

_user_specified_name18699:%!

_user_specified_name18701:%!

_user_specified_name18703:%!

_user_specified_name18705:%!

_user_specified_name18709:% !

_user_specified_name18711:%!!

_user_specified_name18714:%"!

_user_specified_name18716:%#!

_user_specified_name18718:%$!

_user_specified_name18720:%%!

_user_specified_name18724:%&!

_user_specified_name18726:%'!

_user_specified_name18729:%(!

_user_specified_name18731:%)!

_user_specified_name18733:%*!

_user_specified_name18735:%+!

_user_specified_name18739:%,!

_user_specified_name18741:%-!

_user_specified_name18744:%.!

_user_specified_name18746:%/!

_user_specified_name18748:%0!

_user_specified_name18750:%1!

_user_specified_name18754:%2!

_user_specified_name18756:%3!

_user_specified_name18759:%4!

_user_specified_name18761:%5!

_user_specified_name18763:%6!

_user_specified_name18765:%7!

_user_specified_name18769:%8!

_user_specified_name18771:%9!

_user_specified_name18774:%:!

_user_specified_name18776:%;!

_user_specified_name18778:%<!

_user_specified_name18780:%=!

_user_specified_name18784:%>!

_user_specified_name18786:%?!

_user_specified_name18789:%@!

_user_specified_name18791:%A!

_user_specified_name18793:%B!

_user_specified_name18795:%C!

_user_specified_name18799:%D!

_user_specified_name18801
ѕ
V
"__inference__update_step_xla_19466
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19411
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ
J
"__inference__update_step_xla_19456
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
_
C__inference_re_lu_10_layer_call_and_return_conditional_losses_18606

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
–
Ь
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_18249

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
И

–
5__inference_batch_normalization_9_layer_call_fn_20482

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18169Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20472:%!

_user_specified_name20474:%!

_user_specified_name20476:%!

_user_specified_name20478
Ѓ
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_19995

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ж
D
(__inference_re_lu_10_layer_call_fn_20627

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_18606z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Й
њ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20513

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19516
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
≤3
 
%__inference_model_layer_call_fn_19088
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47:@@

unknown_48:@

unknown_49:@

unknown_50:@

unknown_51:@

unknown_52:@$

unknown_53:@@

unknown_54:@

unknown_55:@

unknown_56:@

unknown_57:@

unknown_58:@$

unknown_59:@@

unknown_60:@

unknown_61:@

unknown_62:@

unknown_63:@

unknown_64:@$

unknown_65:@

unknown_66:
identityИҐStatefulPartitionedCallЖ

StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_18806Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapesЄ
µ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:%!

_user_specified_name18950:%!

_user_specified_name18952:%!

_user_specified_name18954:%!

_user_specified_name18956:%!

_user_specified_name18958:%!

_user_specified_name18960:%!

_user_specified_name18962:%!

_user_specified_name18964:%	!

_user_specified_name18966:%
!

_user_specified_name18968:%!

_user_specified_name18970:%!

_user_specified_name18972:%!

_user_specified_name18974:%!

_user_specified_name18976:%!

_user_specified_name18978:%!

_user_specified_name18980:%!

_user_specified_name18982:%!

_user_specified_name18984:%!

_user_specified_name18986:%!

_user_specified_name18988:%!

_user_specified_name18990:%!

_user_specified_name18992:%!

_user_specified_name18994:%!

_user_specified_name18996:%!

_user_specified_name18998:%!

_user_specified_name19000:%!

_user_specified_name19002:%!

_user_specified_name19004:%!

_user_specified_name19006:%!

_user_specified_name19008:%!

_user_specified_name19010:% !

_user_specified_name19012:%!!

_user_specified_name19014:%"!

_user_specified_name19016:%#!

_user_specified_name19018:%$!

_user_specified_name19020:%%!

_user_specified_name19022:%&!

_user_specified_name19024:%'!

_user_specified_name19026:%(!

_user_specified_name19028:%)!

_user_specified_name19030:%*!

_user_specified_name19032:%+!

_user_specified_name19034:%,!

_user_specified_name19036:%-!

_user_specified_name19038:%.!

_user_specified_name19040:%/!

_user_specified_name19042:%0!

_user_specified_name19044:%1!

_user_specified_name19046:%2!

_user_specified_name19048:%3!

_user_specified_name19050:%4!

_user_specified_name19052:%5!

_user_specified_name19054:%6!

_user_specified_name19056:%7!

_user_specified_name19058:%8!

_user_specified_name19060:%9!

_user_specified_name19062:%:!

_user_specified_name19064:%;!

_user_specified_name19066:%<!

_user_specified_name19068:%=!

_user_specified_name19070:%>!

_user_specified_name19072:%?!

_user_specified_name19074:%@!

_user_specified_name19076:%A!

_user_specified_name19078:%B!

_user_specified_name19080:%C!

_user_specified_name19082:%D!

_user_specified_name19084
ѕ
Ы
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_18125

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
±
ь
C__inference_conv2d_8_layer_call_and_return_conditional_losses_18527

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19451
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
V
"__inference__update_step_xla_19486
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ѓ
^
B__inference_re_lu_8_layer_call_and_return_conditional_losses_20450

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_19576
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
Ы
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20076

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ё
Э
(__inference_conv2d_6_layer_call_fn_20186

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_18467Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20180:%!

_user_specified_name20182
И

–
5__inference_batch_normalization_1_layer_call_fn_19754

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17673Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19744:%!

_user_specified_name19746:%!

_user_specified_name19748:%!

_user_specified_name19750
±
ь
C__inference_conv2d_5_layer_call_and_return_conditional_losses_20105

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѕ
Ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19894

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ё
Э
(__inference_conv2d_2_layer_call_fn_19822

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_18347Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19816:%!

_user_specified_name19818
ѕ
V
"__inference__update_step_xla_19526
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
±
ь
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19923

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19491
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
V
"__inference__update_step_xla_19626
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:P L
&
_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
±
ь
C__inference_conv2d_8_layer_call_and_return_conditional_losses_20378

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19611
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѕ
Ы
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20440

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ђ
J
"__inference__update_step_xla_19476
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
цЗ
№o
!__inference__traced_restore_22170
file_prefix8
assignvariableop_conv2d_kernel:@,
assignvariableop_1_conv2d_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@@
2assignvariableop_4_batch_normalization_moving_mean:@D
6assignvariableop_5_batch_normalization_moving_variance:@<
"assignvariableop_6_conv2d_1_kernel:@@.
 assignvariableop_7_conv2d_1_bias:@<
.assignvariableop_8_batch_normalization_1_gamma:@;
-assignvariableop_9_batch_normalization_1_beta:@C
5assignvariableop_10_batch_normalization_1_moving_mean:@G
9assignvariableop_11_batch_normalization_1_moving_variance:@=
#assignvariableop_12_conv2d_2_kernel:@@/
!assignvariableop_13_conv2d_2_bias:@=
/assignvariableop_14_batch_normalization_2_gamma:@<
.assignvariableop_15_batch_normalization_2_beta:@C
5assignvariableop_16_batch_normalization_2_moving_mean:@G
9assignvariableop_17_batch_normalization_2_moving_variance:@=
#assignvariableop_18_conv2d_3_kernel:@@/
!assignvariableop_19_conv2d_3_bias:@=
/assignvariableop_20_batch_normalization_3_gamma:@<
.assignvariableop_21_batch_normalization_3_beta:@C
5assignvariableop_22_batch_normalization_3_moving_mean:@G
9assignvariableop_23_batch_normalization_3_moving_variance:@=
#assignvariableop_24_conv2d_4_kernel:@@/
!assignvariableop_25_conv2d_4_bias:@=
/assignvariableop_26_batch_normalization_4_gamma:@<
.assignvariableop_27_batch_normalization_4_beta:@C
5assignvariableop_28_batch_normalization_4_moving_mean:@G
9assignvariableop_29_batch_normalization_4_moving_variance:@=
#assignvariableop_30_conv2d_5_kernel:@@/
!assignvariableop_31_conv2d_5_bias:@=
/assignvariableop_32_batch_normalization_5_gamma:@<
.assignvariableop_33_batch_normalization_5_beta:@C
5assignvariableop_34_batch_normalization_5_moving_mean:@G
9assignvariableop_35_batch_normalization_5_moving_variance:@=
#assignvariableop_36_conv2d_6_kernel:@@/
!assignvariableop_37_conv2d_6_bias:@=
/assignvariableop_38_batch_normalization_6_gamma:@<
.assignvariableop_39_batch_normalization_6_beta:@C
5assignvariableop_40_batch_normalization_6_moving_mean:@G
9assignvariableop_41_batch_normalization_6_moving_variance:@=
#assignvariableop_42_conv2d_7_kernel:@@/
!assignvariableop_43_conv2d_7_bias:@=
/assignvariableop_44_batch_normalization_7_gamma:@<
.assignvariableop_45_batch_normalization_7_beta:@C
5assignvariableop_46_batch_normalization_7_moving_mean:@G
9assignvariableop_47_batch_normalization_7_moving_variance:@=
#assignvariableop_48_conv2d_8_kernel:@@/
!assignvariableop_49_conv2d_8_bias:@=
/assignvariableop_50_batch_normalization_8_gamma:@<
.assignvariableop_51_batch_normalization_8_beta:@C
5assignvariableop_52_batch_normalization_8_moving_mean:@G
9assignvariableop_53_batch_normalization_8_moving_variance:@=
#assignvariableop_54_conv2d_9_kernel:@@/
!assignvariableop_55_conv2d_9_bias:@=
/assignvariableop_56_batch_normalization_9_gamma:@<
.assignvariableop_57_batch_normalization_9_beta:@C
5assignvariableop_58_batch_normalization_9_moving_mean:@G
9assignvariableop_59_batch_normalization_9_moving_variance:@>
$assignvariableop_60_conv2d_10_kernel:@@0
"assignvariableop_61_conv2d_10_bias:@>
0assignvariableop_62_batch_normalization_10_gamma:@=
/assignvariableop_63_batch_normalization_10_beta:@D
6assignvariableop_64_batch_normalization_10_moving_mean:@H
:assignvariableop_65_batch_normalization_10_moving_variance:@>
$assignvariableop_66_conv2d_11_kernel:@0
"assignvariableop_67_conv2d_11_bias:'
assignvariableop_68_iteration:	 +
!assignvariableop_69_learning_rate: B
(assignvariableop_70_adam_m_conv2d_kernel:@B
(assignvariableop_71_adam_v_conv2d_kernel:@4
&assignvariableop_72_adam_m_conv2d_bias:@4
&assignvariableop_73_adam_v_conv2d_bias:@B
4assignvariableop_74_adam_m_batch_normalization_gamma:@B
4assignvariableop_75_adam_v_batch_normalization_gamma:@A
3assignvariableop_76_adam_m_batch_normalization_beta:@A
3assignvariableop_77_adam_v_batch_normalization_beta:@D
*assignvariableop_78_adam_m_conv2d_1_kernel:@@D
*assignvariableop_79_adam_v_conv2d_1_kernel:@@6
(assignvariableop_80_adam_m_conv2d_1_bias:@6
(assignvariableop_81_adam_v_conv2d_1_bias:@D
6assignvariableop_82_adam_m_batch_normalization_1_gamma:@D
6assignvariableop_83_adam_v_batch_normalization_1_gamma:@C
5assignvariableop_84_adam_m_batch_normalization_1_beta:@C
5assignvariableop_85_adam_v_batch_normalization_1_beta:@D
*assignvariableop_86_adam_m_conv2d_2_kernel:@@D
*assignvariableop_87_adam_v_conv2d_2_kernel:@@6
(assignvariableop_88_adam_m_conv2d_2_bias:@6
(assignvariableop_89_adam_v_conv2d_2_bias:@D
6assignvariableop_90_adam_m_batch_normalization_2_gamma:@D
6assignvariableop_91_adam_v_batch_normalization_2_gamma:@C
5assignvariableop_92_adam_m_batch_normalization_2_beta:@C
5assignvariableop_93_adam_v_batch_normalization_2_beta:@D
*assignvariableop_94_adam_m_conv2d_3_kernel:@@D
*assignvariableop_95_adam_v_conv2d_3_kernel:@@6
(assignvariableop_96_adam_m_conv2d_3_bias:@6
(assignvariableop_97_adam_v_conv2d_3_bias:@D
6assignvariableop_98_adam_m_batch_normalization_3_gamma:@D
6assignvariableop_99_adam_v_batch_normalization_3_gamma:@D
6assignvariableop_100_adam_m_batch_normalization_3_beta:@D
6assignvariableop_101_adam_v_batch_normalization_3_beta:@E
+assignvariableop_102_adam_m_conv2d_4_kernel:@@E
+assignvariableop_103_adam_v_conv2d_4_kernel:@@7
)assignvariableop_104_adam_m_conv2d_4_bias:@7
)assignvariableop_105_adam_v_conv2d_4_bias:@E
7assignvariableop_106_adam_m_batch_normalization_4_gamma:@E
7assignvariableop_107_adam_v_batch_normalization_4_gamma:@D
6assignvariableop_108_adam_m_batch_normalization_4_beta:@D
6assignvariableop_109_adam_v_batch_normalization_4_beta:@E
+assignvariableop_110_adam_m_conv2d_5_kernel:@@E
+assignvariableop_111_adam_v_conv2d_5_kernel:@@7
)assignvariableop_112_adam_m_conv2d_5_bias:@7
)assignvariableop_113_adam_v_conv2d_5_bias:@E
7assignvariableop_114_adam_m_batch_normalization_5_gamma:@E
7assignvariableop_115_adam_v_batch_normalization_5_gamma:@D
6assignvariableop_116_adam_m_batch_normalization_5_beta:@D
6assignvariableop_117_adam_v_batch_normalization_5_beta:@E
+assignvariableop_118_adam_m_conv2d_6_kernel:@@E
+assignvariableop_119_adam_v_conv2d_6_kernel:@@7
)assignvariableop_120_adam_m_conv2d_6_bias:@7
)assignvariableop_121_adam_v_conv2d_6_bias:@E
7assignvariableop_122_adam_m_batch_normalization_6_gamma:@E
7assignvariableop_123_adam_v_batch_normalization_6_gamma:@D
6assignvariableop_124_adam_m_batch_normalization_6_beta:@D
6assignvariableop_125_adam_v_batch_normalization_6_beta:@E
+assignvariableop_126_adam_m_conv2d_7_kernel:@@E
+assignvariableop_127_adam_v_conv2d_7_kernel:@@7
)assignvariableop_128_adam_m_conv2d_7_bias:@7
)assignvariableop_129_adam_v_conv2d_7_bias:@E
7assignvariableop_130_adam_m_batch_normalization_7_gamma:@E
7assignvariableop_131_adam_v_batch_normalization_7_gamma:@D
6assignvariableop_132_adam_m_batch_normalization_7_beta:@D
6assignvariableop_133_adam_v_batch_normalization_7_beta:@E
+assignvariableop_134_adam_m_conv2d_8_kernel:@@E
+assignvariableop_135_adam_v_conv2d_8_kernel:@@7
)assignvariableop_136_adam_m_conv2d_8_bias:@7
)assignvariableop_137_adam_v_conv2d_8_bias:@E
7assignvariableop_138_adam_m_batch_normalization_8_gamma:@E
7assignvariableop_139_adam_v_batch_normalization_8_gamma:@D
6assignvariableop_140_adam_m_batch_normalization_8_beta:@D
6assignvariableop_141_adam_v_batch_normalization_8_beta:@E
+assignvariableop_142_adam_m_conv2d_9_kernel:@@E
+assignvariableop_143_adam_v_conv2d_9_kernel:@@7
)assignvariableop_144_adam_m_conv2d_9_bias:@7
)assignvariableop_145_adam_v_conv2d_9_bias:@E
7assignvariableop_146_adam_m_batch_normalization_9_gamma:@E
7assignvariableop_147_adam_v_batch_normalization_9_gamma:@D
6assignvariableop_148_adam_m_batch_normalization_9_beta:@D
6assignvariableop_149_adam_v_batch_normalization_9_beta:@F
,assignvariableop_150_adam_m_conv2d_10_kernel:@@F
,assignvariableop_151_adam_v_conv2d_10_kernel:@@8
*assignvariableop_152_adam_m_conv2d_10_bias:@8
*assignvariableop_153_adam_v_conv2d_10_bias:@F
8assignvariableop_154_adam_m_batch_normalization_10_gamma:@F
8assignvariableop_155_adam_v_batch_normalization_10_gamma:@E
7assignvariableop_156_adam_m_batch_normalization_10_beta:@E
7assignvariableop_157_adam_v_batch_normalization_10_beta:@F
,assignvariableop_158_adam_m_conv2d_11_kernel:@F
,assignvariableop_159_adam_v_conv2d_11_kernel:@8
*assignvariableop_160_adam_m_conv2d_11_bias:8
*assignvariableop_161_adam_v_conv2d_11_bias:$
assignvariableop_162_total: $
assignvariableop_163_count: 
identity_165ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_102ҐAssignVariableOp_103ҐAssignVariableOp_104ҐAssignVariableOp_105ҐAssignVariableOp_106ҐAssignVariableOp_107ҐAssignVariableOp_108ҐAssignVariableOp_109ҐAssignVariableOp_11ҐAssignVariableOp_110ҐAssignVariableOp_111ҐAssignVariableOp_112ҐAssignVariableOp_113ҐAssignVariableOp_114ҐAssignVariableOp_115ҐAssignVariableOp_116ҐAssignVariableOp_117ҐAssignVariableOp_118ҐAssignVariableOp_119ҐAssignVariableOp_12ҐAssignVariableOp_120ҐAssignVariableOp_121ҐAssignVariableOp_122ҐAssignVariableOp_123ҐAssignVariableOp_124ҐAssignVariableOp_125ҐAssignVariableOp_126ҐAssignVariableOp_127ҐAssignVariableOp_128ҐAssignVariableOp_129ҐAssignVariableOp_13ҐAssignVariableOp_130ҐAssignVariableOp_131ҐAssignVariableOp_132ҐAssignVariableOp_133ҐAssignVariableOp_134ҐAssignVariableOp_135ҐAssignVariableOp_136ҐAssignVariableOp_137ҐAssignVariableOp_138ҐAssignVariableOp_139ҐAssignVariableOp_14ҐAssignVariableOp_140ҐAssignVariableOp_141ҐAssignVariableOp_142ҐAssignVariableOp_143ҐAssignVariableOp_144ҐAssignVariableOp_145ҐAssignVariableOp_146ҐAssignVariableOp_147ҐAssignVariableOp_148ҐAssignVariableOp_149ҐAssignVariableOp_15ҐAssignVariableOp_150ҐAssignVariableOp_151ҐAssignVariableOp_152ҐAssignVariableOp_153ҐAssignVariableOp_154ҐAssignVariableOp_155ҐAssignVariableOp_156ҐAssignVariableOp_157ҐAssignVariableOp_158ҐAssignVariableOp_159ҐAssignVariableOp_16ҐAssignVariableOp_160ҐAssignVariableOp_161ҐAssignVariableOp_162ҐAssignVariableOp_163ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99рF
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:•*
dtype0*ХF
valueЛFBИF•B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHњ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:•*
dtype0*а
value÷B”•B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ё
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*™
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*ґ
dtypesЂ
®2•	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:є
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_5_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_5_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_6_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_6_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_6_betaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_7_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_43AssignVariableOp!assignvariableop_43_conv2d_7_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_44AssignVariableOp/assignvariableop_44_batch_normalization_7_gammaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_45AssignVariableOp.assignvariableop_45_batch_normalization_7_betaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_46AssignVariableOp5assignvariableop_46_batch_normalization_7_moving_meanIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_47AssignVariableOp9assignvariableop_47_batch_normalization_7_moving_varianceIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_48AssignVariableOp#assignvariableop_48_conv2d_8_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_49AssignVariableOp!assignvariableop_49_conv2d_8_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_50AssignVariableOp/assignvariableop_50_batch_normalization_8_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_51AssignVariableOp.assignvariableop_51_batch_normalization_8_betaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_52AssignVariableOp5assignvariableop_52_batch_normalization_8_moving_meanIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_53AssignVariableOp9assignvariableop_53_batch_normalization_8_moving_varianceIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_54AssignVariableOp#assignvariableop_54_conv2d_9_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_55AssignVariableOp!assignvariableop_55_conv2d_9_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_56AssignVariableOp/assignvariableop_56_batch_normalization_9_gammaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_57AssignVariableOp.assignvariableop_57_batch_normalization_9_betaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_58AssignVariableOp5assignvariableop_58_batch_normalization_9_moving_meanIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_59AssignVariableOp9assignvariableop_59_batch_normalization_9_moving_varianceIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_60AssignVariableOp$assignvariableop_60_conv2d_10_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_61AssignVariableOp"assignvariableop_61_conv2d_10_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_62AssignVariableOp0assignvariableop_62_batch_normalization_10_gammaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_63AssignVariableOp/assignvariableop_63_batch_normalization_10_betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_64AssignVariableOp6assignvariableop_64_batch_normalization_10_moving_meanIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_65AssignVariableOp:assignvariableop_65_batch_normalization_10_moving_varianceIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_66AssignVariableOp$assignvariableop_66_conv2d_11_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_67AssignVariableOp"assignvariableop_67_conv2d_11_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_68AssignVariableOpassignvariableop_68_iterationIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_69AssignVariableOp!assignvariableop_69_learning_rateIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_m_conv2d_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_v_conv2d_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_72AssignVariableOp&assignvariableop_72_adam_m_conv2d_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_73AssignVariableOp&assignvariableop_73_adam_v_conv2d_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_74AssignVariableOp4assignvariableop_74_adam_m_batch_normalization_gammaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_75AssignVariableOp4assignvariableop_75_adam_v_batch_normalization_gammaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_76AssignVariableOp3assignvariableop_76_adam_m_batch_normalization_betaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_77AssignVariableOp3assignvariableop_77_adam_v_batch_normalization_betaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_m_conv2d_1_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_v_conv2d_1_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_m_conv2d_1_biasIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adam_v_conv2d_1_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_m_batch_normalization_1_gammaIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_83AssignVariableOp6assignvariableop_83_adam_v_batch_normalization_1_gammaIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_84AssignVariableOp5assignvariableop_84_adam_m_batch_normalization_1_betaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_85AssignVariableOp5assignvariableop_85_adam_v_batch_normalization_1_betaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_m_conv2d_2_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_v_conv2d_2_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_m_conv2d_2_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_89AssignVariableOp(assignvariableop_89_adam_v_conv2d_2_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_m_batch_normalization_2_gammaIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_91AssignVariableOp6assignvariableop_91_adam_v_batch_normalization_2_gammaIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_92AssignVariableOp5assignvariableop_92_adam_m_batch_normalization_2_betaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_93AssignVariableOp5assignvariableop_93_adam_v_batch_normalization_2_betaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_m_conv2d_3_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_v_conv2d_3_kernelIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_m_conv2d_3_biasIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_97AssignVariableOp(assignvariableop_97_adam_v_conv2d_3_biasIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_98AssignVariableOp6assignvariableop_98_adam_m_batch_normalization_3_gammaIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adam_v_batch_normalization_3_gammaIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_100AssignVariableOp6assignvariableop_100_adam_m_batch_normalization_3_betaIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_101AssignVariableOp6assignvariableop_101_adam_v_batch_normalization_3_betaIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_102AssignVariableOp+assignvariableop_102_adam_m_conv2d_4_kernelIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_v_conv2d_4_kernelIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_m_conv2d_4_biasIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_105AssignVariableOp)assignvariableop_105_adam_v_conv2d_4_biasIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_106AssignVariableOp7assignvariableop_106_adam_m_batch_normalization_4_gammaIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_107AssignVariableOp7assignvariableop_107_adam_v_batch_normalization_4_gammaIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_108AssignVariableOp6assignvariableop_108_adam_m_batch_normalization_4_betaIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_109AssignVariableOp6assignvariableop_109_adam_v_batch_normalization_4_betaIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_110AssignVariableOp+assignvariableop_110_adam_m_conv2d_5_kernelIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_v_conv2d_5_kernelIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_m_conv2d_5_biasIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_113AssignVariableOp)assignvariableop_113_adam_v_conv2d_5_biasIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_114AssignVariableOp7assignvariableop_114_adam_m_batch_normalization_5_gammaIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_115AssignVariableOp7assignvariableop_115_adam_v_batch_normalization_5_gammaIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_116AssignVariableOp6assignvariableop_116_adam_m_batch_normalization_5_betaIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_117AssignVariableOp6assignvariableop_117_adam_v_batch_normalization_5_betaIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_118AssignVariableOp+assignvariableop_118_adam_m_conv2d_6_kernelIdentity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_119AssignVariableOp+assignvariableop_119_adam_v_conv2d_6_kernelIdentity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_120AssignVariableOp)assignvariableop_120_adam_m_conv2d_6_biasIdentity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_121AssignVariableOp)assignvariableop_121_adam_v_conv2d_6_biasIdentity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_122AssignVariableOp7assignvariableop_122_adam_m_batch_normalization_6_gammaIdentity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_123AssignVariableOp7assignvariableop_123_adam_v_batch_normalization_6_gammaIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_124AssignVariableOp6assignvariableop_124_adam_m_batch_normalization_6_betaIdentity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_125AssignVariableOp6assignvariableop_125_adam_v_batch_normalization_6_betaIdentity_125:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_126AssignVariableOp+assignvariableop_126_adam_m_conv2d_7_kernelIdentity_126:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_127AssignVariableOp+assignvariableop_127_adam_v_conv2d_7_kernelIdentity_127:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_128AssignVariableOp)assignvariableop_128_adam_m_conv2d_7_biasIdentity_128:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_129AssignVariableOp)assignvariableop_129_adam_v_conv2d_7_biasIdentity_129:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_130AssignVariableOp7assignvariableop_130_adam_m_batch_normalization_7_gammaIdentity_130:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_131AssignVariableOp7assignvariableop_131_adam_v_batch_normalization_7_gammaIdentity_131:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_132AssignVariableOp6assignvariableop_132_adam_m_batch_normalization_7_betaIdentity_132:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_133AssignVariableOp6assignvariableop_133_adam_v_batch_normalization_7_betaIdentity_133:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_134AssignVariableOp+assignvariableop_134_adam_m_conv2d_8_kernelIdentity_134:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_135AssignVariableOp+assignvariableop_135_adam_v_conv2d_8_kernelIdentity_135:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_136AssignVariableOp)assignvariableop_136_adam_m_conv2d_8_biasIdentity_136:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_137AssignVariableOp)assignvariableop_137_adam_v_conv2d_8_biasIdentity_137:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_138AssignVariableOp7assignvariableop_138_adam_m_batch_normalization_8_gammaIdentity_138:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_139AssignVariableOp7assignvariableop_139_adam_v_batch_normalization_8_gammaIdentity_139:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_140AssignVariableOp6assignvariableop_140_adam_m_batch_normalization_8_betaIdentity_140:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_141AssignVariableOp6assignvariableop_141_adam_v_batch_normalization_8_betaIdentity_141:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_142AssignVariableOp+assignvariableop_142_adam_m_conv2d_9_kernelIdentity_142:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_143AssignVariableOp+assignvariableop_143_adam_v_conv2d_9_kernelIdentity_143:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_144AssignVariableOp)assignvariableop_144_adam_m_conv2d_9_biasIdentity_144:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_145AssignVariableOp)assignvariableop_145_adam_v_conv2d_9_biasIdentity_145:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_146AssignVariableOp7assignvariableop_146_adam_m_batch_normalization_9_gammaIdentity_146:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_147AssignVariableOp7assignvariableop_147_adam_v_batch_normalization_9_gammaIdentity_147:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_148AssignVariableOp6assignvariableop_148_adam_m_batch_normalization_9_betaIdentity_148:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_149AssignVariableOp6assignvariableop_149_adam_v_batch_normalization_9_betaIdentity_149:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_150AssignVariableOp,assignvariableop_150_adam_m_conv2d_10_kernelIdentity_150:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_151AssignVariableOp,assignvariableop_151_adam_v_conv2d_10_kernelIdentity_151:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_152AssignVariableOp*assignvariableop_152_adam_m_conv2d_10_biasIdentity_152:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_153AssignVariableOp*assignvariableop_153_adam_v_conv2d_10_biasIdentity_153:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_154AssignVariableOp8assignvariableop_154_adam_m_batch_normalization_10_gammaIdentity_154:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_155AssignVariableOp8assignvariableop_155_adam_v_batch_normalization_10_gammaIdentity_155:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_156AssignVariableOp7assignvariableop_156_adam_m_batch_normalization_10_betaIdentity_156:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_157AssignVariableOp7assignvariableop_157_adam_v_batch_normalization_10_betaIdentity_157:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_158AssignVariableOp,assignvariableop_158_adam_m_conv2d_11_kernelIdentity_158:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_159AssignVariableOp,assignvariableop_159_adam_v_conv2d_11_kernelIdentity_159:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_160AssignVariableOp*assignvariableop_160_adam_m_conv2d_11_biasIdentity_160:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_161AssignVariableOp*assignvariableop_161_adam_v_conv2d_11_biasIdentity_161:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_162AssignVariableOpassignvariableop_162_totalIdentity_162:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_163AssignVariableOpassignvariableop_163_countIdentity_163:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ®
Identity_164Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_165IdentityIdentity_164:output:0^NoOp_1*
T0*
_output_shapes
: р
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_165Identity_165:output:0*(
_construction_contextkEagerRuntime*я
_input_shapesЌ
 : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv2d/kernel:+'
%
_user_specified_nameconv2d/bias:95
3
_user_specified_namebatch_normalization/gamma:84
2
_user_specified_namebatch_normalization/beta:?;
9
_user_specified_name!batch_normalization/moving_mean:C?
=
_user_specified_name%#batch_normalization/moving_variance:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv2d_1/bias:;	7
5
_user_specified_namebatch_normalization_1/gamma::
6
4
_user_specified_namebatch_normalization_1/beta:A=
;
_user_specified_name#!batch_normalization_1/moving_mean:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:/+
)
_user_specified_nameconv2d_2/kernel:-)
'
_user_specified_nameconv2d_2/bias:;7
5
_user_specified_namebatch_normalization_2/gamma::6
4
_user_specified_namebatch_normalization_2/beta:A=
;
_user_specified_name#!batch_normalization_2/moving_mean:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_3/bias:;7
5
_user_specified_namebatch_normalization_3/gamma::6
4
_user_specified_namebatch_normalization_3/beta:A=
;
_user_specified_name#!batch_normalization_3/moving_mean:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:;7
5
_user_specified_namebatch_normalization_4/gamma::6
4
_user_specified_namebatch_normalization_4/beta:A=
;
_user_specified_name#!batch_normalization_4/moving_mean:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:/+
)
_user_specified_nameconv2d_5/kernel:- )
'
_user_specified_nameconv2d_5/bias:;!7
5
_user_specified_namebatch_normalization_5/gamma::"6
4
_user_specified_namebatch_normalization_5/beta:A#=
;
_user_specified_name#!batch_normalization_5/moving_mean:E$A
?
_user_specified_name'%batch_normalization_5/moving_variance:/%+
)
_user_specified_nameconv2d_6/kernel:-&)
'
_user_specified_nameconv2d_6/bias:;'7
5
_user_specified_namebatch_normalization_6/gamma::(6
4
_user_specified_namebatch_normalization_6/beta:A)=
;
_user_specified_name#!batch_normalization_6/moving_mean:E*A
?
_user_specified_name'%batch_normalization_6/moving_variance:/++
)
_user_specified_nameconv2d_7/kernel:-,)
'
_user_specified_nameconv2d_7/bias:;-7
5
_user_specified_namebatch_normalization_7/gamma::.6
4
_user_specified_namebatch_normalization_7/beta:A/=
;
_user_specified_name#!batch_normalization_7/moving_mean:E0A
?
_user_specified_name'%batch_normalization_7/moving_variance:/1+
)
_user_specified_nameconv2d_8/kernel:-2)
'
_user_specified_nameconv2d_8/bias:;37
5
_user_specified_namebatch_normalization_8/gamma::46
4
_user_specified_namebatch_normalization_8/beta:A5=
;
_user_specified_name#!batch_normalization_8/moving_mean:E6A
?
_user_specified_name'%batch_normalization_8/moving_variance:/7+
)
_user_specified_nameconv2d_9/kernel:-8)
'
_user_specified_nameconv2d_9/bias:;97
5
_user_specified_namebatch_normalization_9/gamma:::6
4
_user_specified_namebatch_normalization_9/beta:A;=
;
_user_specified_name#!batch_normalization_9/moving_mean:E<A
?
_user_specified_name'%batch_normalization_9/moving_variance:0=,
*
_user_specified_nameconv2d_10/kernel:.>*
(
_user_specified_nameconv2d_10/bias:<?8
6
_user_specified_namebatch_normalization_10/gamma:;@7
5
_user_specified_namebatch_normalization_10/beta:BA>
<
_user_specified_name$"batch_normalization_10/moving_mean:FBB
@
_user_specified_name(&batch_normalization_10/moving_variance:0C,
*
_user_specified_nameconv2d_11/kernel:.D*
(
_user_specified_nameconv2d_11/bias:)E%
#
_user_specified_name	iteration:-F)
'
_user_specified_namelearning_rate:4G0
.
_user_specified_nameAdam/m/conv2d/kernel:4H0
.
_user_specified_nameAdam/v/conv2d/kernel:2I.
,
_user_specified_nameAdam/m/conv2d/bias:2J.
,
_user_specified_nameAdam/v/conv2d/bias:@K<
:
_user_specified_name" Adam/m/batch_normalization/gamma:@L<
:
_user_specified_name" Adam/v/batch_normalization/gamma:?M;
9
_user_specified_name!Adam/m/batch_normalization/beta:?N;
9
_user_specified_name!Adam/v/batch_normalization/beta:6O2
0
_user_specified_nameAdam/m/conv2d_1/kernel:6P2
0
_user_specified_nameAdam/v/conv2d_1/kernel:4Q0
.
_user_specified_nameAdam/m/conv2d_1/bias:4R0
.
_user_specified_nameAdam/v/conv2d_1/bias:BS>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:BT>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:AU=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:AV=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:6W2
0
_user_specified_nameAdam/m/conv2d_2/kernel:6X2
0
_user_specified_nameAdam/v/conv2d_2/kernel:4Y0
.
_user_specified_nameAdam/m/conv2d_2/bias:4Z0
.
_user_specified_nameAdam/v/conv2d_2/bias:B[>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:B\>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:A]=
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:A^=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:6_2
0
_user_specified_nameAdam/m/conv2d_3/kernel:6`2
0
_user_specified_nameAdam/v/conv2d_3/kernel:4a0
.
_user_specified_nameAdam/m/conv2d_3/bias:4b0
.
_user_specified_nameAdam/v/conv2d_3/bias:Bc>
<
_user_specified_name$"Adam/m/batch_normalization_3/gamma:Bd>
<
_user_specified_name$"Adam/v/batch_normalization_3/gamma:Ae=
;
_user_specified_name#!Adam/m/batch_normalization_3/beta:Af=
;
_user_specified_name#!Adam/v/batch_normalization_3/beta:6g2
0
_user_specified_nameAdam/m/conv2d_4/kernel:6h2
0
_user_specified_nameAdam/v/conv2d_4/kernel:4i0
.
_user_specified_nameAdam/m/conv2d_4/bias:4j0
.
_user_specified_nameAdam/v/conv2d_4/bias:Bk>
<
_user_specified_name$"Adam/m/batch_normalization_4/gamma:Bl>
<
_user_specified_name$"Adam/v/batch_normalization_4/gamma:Am=
;
_user_specified_name#!Adam/m/batch_normalization_4/beta:An=
;
_user_specified_name#!Adam/v/batch_normalization_4/beta:6o2
0
_user_specified_nameAdam/m/conv2d_5/kernel:6p2
0
_user_specified_nameAdam/v/conv2d_5/kernel:4q0
.
_user_specified_nameAdam/m/conv2d_5/bias:4r0
.
_user_specified_nameAdam/v/conv2d_5/bias:Bs>
<
_user_specified_name$"Adam/m/batch_normalization_5/gamma:Bt>
<
_user_specified_name$"Adam/v/batch_normalization_5/gamma:Au=
;
_user_specified_name#!Adam/m/batch_normalization_5/beta:Av=
;
_user_specified_name#!Adam/v/batch_normalization_5/beta:6w2
0
_user_specified_nameAdam/m/conv2d_6/kernel:6x2
0
_user_specified_nameAdam/v/conv2d_6/kernel:4y0
.
_user_specified_nameAdam/m/conv2d_6/bias:4z0
.
_user_specified_nameAdam/v/conv2d_6/bias:B{>
<
_user_specified_name$"Adam/m/batch_normalization_6/gamma:B|>
<
_user_specified_name$"Adam/v/batch_normalization_6/gamma:A}=
;
_user_specified_name#!Adam/m/batch_normalization_6/beta:A~=
;
_user_specified_name#!Adam/v/batch_normalization_6/beta:62
0
_user_specified_nameAdam/m/conv2d_7/kernel:7А2
0
_user_specified_nameAdam/v/conv2d_7/kernel:5Б0
.
_user_specified_nameAdam/m/conv2d_7/bias:5В0
.
_user_specified_nameAdam/v/conv2d_7/bias:CГ>
<
_user_specified_name$"Adam/m/batch_normalization_7/gamma:CД>
<
_user_specified_name$"Adam/v/batch_normalization_7/gamma:BЕ=
;
_user_specified_name#!Adam/m/batch_normalization_7/beta:BЖ=
;
_user_specified_name#!Adam/v/batch_normalization_7/beta:7З2
0
_user_specified_nameAdam/m/conv2d_8/kernel:7И2
0
_user_specified_nameAdam/v/conv2d_8/kernel:5Й0
.
_user_specified_nameAdam/m/conv2d_8/bias:5К0
.
_user_specified_nameAdam/v/conv2d_8/bias:CЛ>
<
_user_specified_name$"Adam/m/batch_normalization_8/gamma:CМ>
<
_user_specified_name$"Adam/v/batch_normalization_8/gamma:BН=
;
_user_specified_name#!Adam/m/batch_normalization_8/beta:BО=
;
_user_specified_name#!Adam/v/batch_normalization_8/beta:7П2
0
_user_specified_nameAdam/m/conv2d_9/kernel:7Р2
0
_user_specified_nameAdam/v/conv2d_9/kernel:5С0
.
_user_specified_nameAdam/m/conv2d_9/bias:5Т0
.
_user_specified_nameAdam/v/conv2d_9/bias:CУ>
<
_user_specified_name$"Adam/m/batch_normalization_9/gamma:CФ>
<
_user_specified_name$"Adam/v/batch_normalization_9/gamma:BХ=
;
_user_specified_name#!Adam/m/batch_normalization_9/beta:BЦ=
;
_user_specified_name#!Adam/v/batch_normalization_9/beta:8Ч3
1
_user_specified_nameAdam/m/conv2d_10/kernel:8Ш3
1
_user_specified_nameAdam/v/conv2d_10/kernel:6Щ1
/
_user_specified_nameAdam/m/conv2d_10/bias:6Ъ1
/
_user_specified_nameAdam/v/conv2d_10/bias:DЫ?
=
_user_specified_name%#Adam/m/batch_normalization_10/gamma:DЬ?
=
_user_specified_name%#Adam/v/batch_normalization_10/gamma:CЭ>
<
_user_specified_name$"Adam/m/batch_normalization_10/beta:CЮ>
<
_user_specified_name$"Adam/v/batch_normalization_10/beta:8Я3
1
_user_specified_nameAdam/m/conv2d_11/kernel:8†3
1
_user_specified_nameAdam/v/conv2d_11/kernel:6°1
/
_user_specified_nameAdam/m/conv2d_11/bias:6Ґ1
/
_user_specified_nameAdam/v/conv2d_11/bias:&£!

_user_specified_nametotal:&§!

_user_specified_namecount
ѕ
Ы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_18001

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Д
C
'__inference_re_lu_7_layer_call_fn_20354

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_18516z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ё
Э
(__inference_conv2d_9_layer_call_fn_20459

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_18557Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20453:%!

_user_specified_name20455
ѕ
V
"__inference__update_step_xla_19566
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
К

–
5__inference_batch_normalization_9_layer_call_fn_20495

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18187Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20485:%!

_user_specified_name20487:%!

_user_specified_name20489:%!

_user_specified_name20491
Д
C
'__inference_re_lu_8_layer_call_fn_20445

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_18546z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
µ
O
#__inference_add_layer_call_fn_20657
inputs_0
inputs_1
identity”
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_18628z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_1
Ђ
J
"__inference__update_step_xla_19631
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
К

—
6__inference_batch_normalization_10_layer_call_fn_20573

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_18231Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20563:%!

_user_specified_name20565:%!

_user_specified_name20567:%!

_user_specified_name20569
Й
њ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_17983

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
±
ь
C__inference_conv2d_7_layer_call_and_return_conditional_losses_20287

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ђ
\
@__inference_re_lu_layer_call_and_return_conditional_losses_19722

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
„
j
>__inference_add_layer_call_and_return_conditional_losses_20663
inputs_0
inputs_1
identityl
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_1
Љи	
иЯ
__inference__traced_save_21669
file_prefix>
$read_disablecopyonread_conv2d_kernel:@2
$read_1_disablecopyonread_conv2d_bias:@@
2read_2_disablecopyonread_batch_normalization_gamma:@?
1read_3_disablecopyonread_batch_normalization_beta:@F
8read_4_disablecopyonread_batch_normalization_moving_mean:@J
<read_5_disablecopyonread_batch_normalization_moving_variance:@B
(read_6_disablecopyonread_conv2d_1_kernel:@@4
&read_7_disablecopyonread_conv2d_1_bias:@B
4read_8_disablecopyonread_batch_normalization_1_gamma:@A
3read_9_disablecopyonread_batch_normalization_1_beta:@I
;read_10_disablecopyonread_batch_normalization_1_moving_mean:@M
?read_11_disablecopyonread_batch_normalization_1_moving_variance:@C
)read_12_disablecopyonread_conv2d_2_kernel:@@5
'read_13_disablecopyonread_conv2d_2_bias:@C
5read_14_disablecopyonread_batch_normalization_2_gamma:@B
4read_15_disablecopyonread_batch_normalization_2_beta:@I
;read_16_disablecopyonread_batch_normalization_2_moving_mean:@M
?read_17_disablecopyonread_batch_normalization_2_moving_variance:@C
)read_18_disablecopyonread_conv2d_3_kernel:@@5
'read_19_disablecopyonread_conv2d_3_bias:@C
5read_20_disablecopyonread_batch_normalization_3_gamma:@B
4read_21_disablecopyonread_batch_normalization_3_beta:@I
;read_22_disablecopyonread_batch_normalization_3_moving_mean:@M
?read_23_disablecopyonread_batch_normalization_3_moving_variance:@C
)read_24_disablecopyonread_conv2d_4_kernel:@@5
'read_25_disablecopyonread_conv2d_4_bias:@C
5read_26_disablecopyonread_batch_normalization_4_gamma:@B
4read_27_disablecopyonread_batch_normalization_4_beta:@I
;read_28_disablecopyonread_batch_normalization_4_moving_mean:@M
?read_29_disablecopyonread_batch_normalization_4_moving_variance:@C
)read_30_disablecopyonread_conv2d_5_kernel:@@5
'read_31_disablecopyonread_conv2d_5_bias:@C
5read_32_disablecopyonread_batch_normalization_5_gamma:@B
4read_33_disablecopyonread_batch_normalization_5_beta:@I
;read_34_disablecopyonread_batch_normalization_5_moving_mean:@M
?read_35_disablecopyonread_batch_normalization_5_moving_variance:@C
)read_36_disablecopyonread_conv2d_6_kernel:@@5
'read_37_disablecopyonread_conv2d_6_bias:@C
5read_38_disablecopyonread_batch_normalization_6_gamma:@B
4read_39_disablecopyonread_batch_normalization_6_beta:@I
;read_40_disablecopyonread_batch_normalization_6_moving_mean:@M
?read_41_disablecopyonread_batch_normalization_6_moving_variance:@C
)read_42_disablecopyonread_conv2d_7_kernel:@@5
'read_43_disablecopyonread_conv2d_7_bias:@C
5read_44_disablecopyonread_batch_normalization_7_gamma:@B
4read_45_disablecopyonread_batch_normalization_7_beta:@I
;read_46_disablecopyonread_batch_normalization_7_moving_mean:@M
?read_47_disablecopyonread_batch_normalization_7_moving_variance:@C
)read_48_disablecopyonread_conv2d_8_kernel:@@5
'read_49_disablecopyonread_conv2d_8_bias:@C
5read_50_disablecopyonread_batch_normalization_8_gamma:@B
4read_51_disablecopyonread_batch_normalization_8_beta:@I
;read_52_disablecopyonread_batch_normalization_8_moving_mean:@M
?read_53_disablecopyonread_batch_normalization_8_moving_variance:@C
)read_54_disablecopyonread_conv2d_9_kernel:@@5
'read_55_disablecopyonread_conv2d_9_bias:@C
5read_56_disablecopyonread_batch_normalization_9_gamma:@B
4read_57_disablecopyonread_batch_normalization_9_beta:@I
;read_58_disablecopyonread_batch_normalization_9_moving_mean:@M
?read_59_disablecopyonread_batch_normalization_9_moving_variance:@D
*read_60_disablecopyonread_conv2d_10_kernel:@@6
(read_61_disablecopyonread_conv2d_10_bias:@D
6read_62_disablecopyonread_batch_normalization_10_gamma:@C
5read_63_disablecopyonread_batch_normalization_10_beta:@J
<read_64_disablecopyonread_batch_normalization_10_moving_mean:@N
@read_65_disablecopyonread_batch_normalization_10_moving_variance:@D
*read_66_disablecopyonread_conv2d_11_kernel:@6
(read_67_disablecopyonread_conv2d_11_bias:-
#read_68_disablecopyonread_iteration:	 1
'read_69_disablecopyonread_learning_rate: H
.read_70_disablecopyonread_adam_m_conv2d_kernel:@H
.read_71_disablecopyonread_adam_v_conv2d_kernel:@:
,read_72_disablecopyonread_adam_m_conv2d_bias:@:
,read_73_disablecopyonread_adam_v_conv2d_bias:@H
:read_74_disablecopyonread_adam_m_batch_normalization_gamma:@H
:read_75_disablecopyonread_adam_v_batch_normalization_gamma:@G
9read_76_disablecopyonread_adam_m_batch_normalization_beta:@G
9read_77_disablecopyonread_adam_v_batch_normalization_beta:@J
0read_78_disablecopyonread_adam_m_conv2d_1_kernel:@@J
0read_79_disablecopyonread_adam_v_conv2d_1_kernel:@@<
.read_80_disablecopyonread_adam_m_conv2d_1_bias:@<
.read_81_disablecopyonread_adam_v_conv2d_1_bias:@J
<read_82_disablecopyonread_adam_m_batch_normalization_1_gamma:@J
<read_83_disablecopyonread_adam_v_batch_normalization_1_gamma:@I
;read_84_disablecopyonread_adam_m_batch_normalization_1_beta:@I
;read_85_disablecopyonread_adam_v_batch_normalization_1_beta:@J
0read_86_disablecopyonread_adam_m_conv2d_2_kernel:@@J
0read_87_disablecopyonread_adam_v_conv2d_2_kernel:@@<
.read_88_disablecopyonread_adam_m_conv2d_2_bias:@<
.read_89_disablecopyonread_adam_v_conv2d_2_bias:@J
<read_90_disablecopyonread_adam_m_batch_normalization_2_gamma:@J
<read_91_disablecopyonread_adam_v_batch_normalization_2_gamma:@I
;read_92_disablecopyonread_adam_m_batch_normalization_2_beta:@I
;read_93_disablecopyonread_adam_v_batch_normalization_2_beta:@J
0read_94_disablecopyonread_adam_m_conv2d_3_kernel:@@J
0read_95_disablecopyonread_adam_v_conv2d_3_kernel:@@<
.read_96_disablecopyonread_adam_m_conv2d_3_bias:@<
.read_97_disablecopyonread_adam_v_conv2d_3_bias:@J
<read_98_disablecopyonread_adam_m_batch_normalization_3_gamma:@J
<read_99_disablecopyonread_adam_v_batch_normalization_3_gamma:@J
<read_100_disablecopyonread_adam_m_batch_normalization_3_beta:@J
<read_101_disablecopyonread_adam_v_batch_normalization_3_beta:@K
1read_102_disablecopyonread_adam_m_conv2d_4_kernel:@@K
1read_103_disablecopyonread_adam_v_conv2d_4_kernel:@@=
/read_104_disablecopyonread_adam_m_conv2d_4_bias:@=
/read_105_disablecopyonread_adam_v_conv2d_4_bias:@K
=read_106_disablecopyonread_adam_m_batch_normalization_4_gamma:@K
=read_107_disablecopyonread_adam_v_batch_normalization_4_gamma:@J
<read_108_disablecopyonread_adam_m_batch_normalization_4_beta:@J
<read_109_disablecopyonread_adam_v_batch_normalization_4_beta:@K
1read_110_disablecopyonread_adam_m_conv2d_5_kernel:@@K
1read_111_disablecopyonread_adam_v_conv2d_5_kernel:@@=
/read_112_disablecopyonread_adam_m_conv2d_5_bias:@=
/read_113_disablecopyonread_adam_v_conv2d_5_bias:@K
=read_114_disablecopyonread_adam_m_batch_normalization_5_gamma:@K
=read_115_disablecopyonread_adam_v_batch_normalization_5_gamma:@J
<read_116_disablecopyonread_adam_m_batch_normalization_5_beta:@J
<read_117_disablecopyonread_adam_v_batch_normalization_5_beta:@K
1read_118_disablecopyonread_adam_m_conv2d_6_kernel:@@K
1read_119_disablecopyonread_adam_v_conv2d_6_kernel:@@=
/read_120_disablecopyonread_adam_m_conv2d_6_bias:@=
/read_121_disablecopyonread_adam_v_conv2d_6_bias:@K
=read_122_disablecopyonread_adam_m_batch_normalization_6_gamma:@K
=read_123_disablecopyonread_adam_v_batch_normalization_6_gamma:@J
<read_124_disablecopyonread_adam_m_batch_normalization_6_beta:@J
<read_125_disablecopyonread_adam_v_batch_normalization_6_beta:@K
1read_126_disablecopyonread_adam_m_conv2d_7_kernel:@@K
1read_127_disablecopyonread_adam_v_conv2d_7_kernel:@@=
/read_128_disablecopyonread_adam_m_conv2d_7_bias:@=
/read_129_disablecopyonread_adam_v_conv2d_7_bias:@K
=read_130_disablecopyonread_adam_m_batch_normalization_7_gamma:@K
=read_131_disablecopyonread_adam_v_batch_normalization_7_gamma:@J
<read_132_disablecopyonread_adam_m_batch_normalization_7_beta:@J
<read_133_disablecopyonread_adam_v_batch_normalization_7_beta:@K
1read_134_disablecopyonread_adam_m_conv2d_8_kernel:@@K
1read_135_disablecopyonread_adam_v_conv2d_8_kernel:@@=
/read_136_disablecopyonread_adam_m_conv2d_8_bias:@=
/read_137_disablecopyonread_adam_v_conv2d_8_bias:@K
=read_138_disablecopyonread_adam_m_batch_normalization_8_gamma:@K
=read_139_disablecopyonread_adam_v_batch_normalization_8_gamma:@J
<read_140_disablecopyonread_adam_m_batch_normalization_8_beta:@J
<read_141_disablecopyonread_adam_v_batch_normalization_8_beta:@K
1read_142_disablecopyonread_adam_m_conv2d_9_kernel:@@K
1read_143_disablecopyonread_adam_v_conv2d_9_kernel:@@=
/read_144_disablecopyonread_adam_m_conv2d_9_bias:@=
/read_145_disablecopyonread_adam_v_conv2d_9_bias:@K
=read_146_disablecopyonread_adam_m_batch_normalization_9_gamma:@K
=read_147_disablecopyonread_adam_v_batch_normalization_9_gamma:@J
<read_148_disablecopyonread_adam_m_batch_normalization_9_beta:@J
<read_149_disablecopyonread_adam_v_batch_normalization_9_beta:@L
2read_150_disablecopyonread_adam_m_conv2d_10_kernel:@@L
2read_151_disablecopyonread_adam_v_conv2d_10_kernel:@@>
0read_152_disablecopyonread_adam_m_conv2d_10_bias:@>
0read_153_disablecopyonread_adam_v_conv2d_10_bias:@L
>read_154_disablecopyonread_adam_m_batch_normalization_10_gamma:@L
>read_155_disablecopyonread_adam_v_batch_normalization_10_gamma:@K
=read_156_disablecopyonread_adam_m_batch_normalization_10_beta:@K
=read_157_disablecopyonread_adam_v_batch_normalization_10_beta:@L
2read_158_disablecopyonread_adam_m_conv2d_11_kernel:@L
2read_159_disablecopyonread_adam_v_conv2d_11_kernel:@>
0read_160_disablecopyonread_adam_m_conv2d_11_bias:>
0read_161_disablecopyonread_adam_v_conv2d_11_bias:*
 read_162_disablecopyonread_total: *
 read_163_disablecopyonread_count: 
savev2_const
identity_329ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_100/DisableCopyOnReadҐRead_100/ReadVariableOpҐRead_101/DisableCopyOnReadҐRead_101/ReadVariableOpҐRead_102/DisableCopyOnReadҐRead_102/ReadVariableOpҐRead_103/DisableCopyOnReadҐRead_103/ReadVariableOpҐRead_104/DisableCopyOnReadҐRead_104/ReadVariableOpҐRead_105/DisableCopyOnReadҐRead_105/ReadVariableOpҐRead_106/DisableCopyOnReadҐRead_106/ReadVariableOpҐRead_107/DisableCopyOnReadҐRead_107/ReadVariableOpҐRead_108/DisableCopyOnReadҐRead_108/ReadVariableOpҐRead_109/DisableCopyOnReadҐRead_109/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_110/DisableCopyOnReadҐRead_110/ReadVariableOpҐRead_111/DisableCopyOnReadҐRead_111/ReadVariableOpҐRead_112/DisableCopyOnReadҐRead_112/ReadVariableOpҐRead_113/DisableCopyOnReadҐRead_113/ReadVariableOpҐRead_114/DisableCopyOnReadҐRead_114/ReadVariableOpҐRead_115/DisableCopyOnReadҐRead_115/ReadVariableOpҐRead_116/DisableCopyOnReadҐRead_116/ReadVariableOpҐRead_117/DisableCopyOnReadҐRead_117/ReadVariableOpҐRead_118/DisableCopyOnReadҐRead_118/ReadVariableOpҐRead_119/DisableCopyOnReadҐRead_119/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_120/DisableCopyOnReadҐRead_120/ReadVariableOpҐRead_121/DisableCopyOnReadҐRead_121/ReadVariableOpҐRead_122/DisableCopyOnReadҐRead_122/ReadVariableOpҐRead_123/DisableCopyOnReadҐRead_123/ReadVariableOpҐRead_124/DisableCopyOnReadҐRead_124/ReadVariableOpҐRead_125/DisableCopyOnReadҐRead_125/ReadVariableOpҐRead_126/DisableCopyOnReadҐRead_126/ReadVariableOpҐRead_127/DisableCopyOnReadҐRead_127/ReadVariableOpҐRead_128/DisableCopyOnReadҐRead_128/ReadVariableOpҐRead_129/DisableCopyOnReadҐRead_129/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_130/DisableCopyOnReadҐRead_130/ReadVariableOpҐRead_131/DisableCopyOnReadҐRead_131/ReadVariableOpҐRead_132/DisableCopyOnReadҐRead_132/ReadVariableOpҐRead_133/DisableCopyOnReadҐRead_133/ReadVariableOpҐRead_134/DisableCopyOnReadҐRead_134/ReadVariableOpҐRead_135/DisableCopyOnReadҐRead_135/ReadVariableOpҐRead_136/DisableCopyOnReadҐRead_136/ReadVariableOpҐRead_137/DisableCopyOnReadҐRead_137/ReadVariableOpҐRead_138/DisableCopyOnReadҐRead_138/ReadVariableOpҐRead_139/DisableCopyOnReadҐRead_139/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_140/DisableCopyOnReadҐRead_140/ReadVariableOpҐRead_141/DisableCopyOnReadҐRead_141/ReadVariableOpҐRead_142/DisableCopyOnReadҐRead_142/ReadVariableOpҐRead_143/DisableCopyOnReadҐRead_143/ReadVariableOpҐRead_144/DisableCopyOnReadҐRead_144/ReadVariableOpҐRead_145/DisableCopyOnReadҐRead_145/ReadVariableOpҐRead_146/DisableCopyOnReadҐRead_146/ReadVariableOpҐRead_147/DisableCopyOnReadҐRead_147/ReadVariableOpҐRead_148/DisableCopyOnReadҐRead_148/ReadVariableOpҐRead_149/DisableCopyOnReadҐRead_149/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_150/DisableCopyOnReadҐRead_150/ReadVariableOpҐRead_151/DisableCopyOnReadҐRead_151/ReadVariableOpҐRead_152/DisableCopyOnReadҐRead_152/ReadVariableOpҐRead_153/DisableCopyOnReadҐRead_153/ReadVariableOpҐRead_154/DisableCopyOnReadҐRead_154/ReadVariableOpҐRead_155/DisableCopyOnReadҐRead_155/ReadVariableOpҐRead_156/DisableCopyOnReadҐRead_156/ReadVariableOpҐRead_157/DisableCopyOnReadҐRead_157/ReadVariableOpҐRead_158/DisableCopyOnReadҐRead_158/ReadVariableOpҐRead_159/DisableCopyOnReadҐRead_159/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_160/DisableCopyOnReadҐRead_160/ReadVariableOpҐRead_161/DisableCopyOnReadҐRead_161/ReadVariableOpҐRead_162/DisableCopyOnReadҐRead_162/ReadVariableOpҐRead_163/DisableCopyOnReadҐRead_163/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_28/DisableCopyOnReadҐRead_28/ReadVariableOpҐRead_29/DisableCopyOnReadҐRead_29/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_30/DisableCopyOnReadҐRead_30/ReadVariableOpҐRead_31/DisableCopyOnReadҐRead_31/ReadVariableOpҐRead_32/DisableCopyOnReadҐRead_32/ReadVariableOpҐRead_33/DisableCopyOnReadҐRead_33/ReadVariableOpҐRead_34/DisableCopyOnReadҐRead_34/ReadVariableOpҐRead_35/DisableCopyOnReadҐRead_35/ReadVariableOpҐRead_36/DisableCopyOnReadҐRead_36/ReadVariableOpҐRead_37/DisableCopyOnReadҐRead_37/ReadVariableOpҐRead_38/DisableCopyOnReadҐRead_38/ReadVariableOpҐRead_39/DisableCopyOnReadҐRead_39/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_40/DisableCopyOnReadҐRead_40/ReadVariableOpҐRead_41/DisableCopyOnReadҐRead_41/ReadVariableOpҐRead_42/DisableCopyOnReadҐRead_42/ReadVariableOpҐRead_43/DisableCopyOnReadҐRead_43/ReadVariableOpҐRead_44/DisableCopyOnReadҐRead_44/ReadVariableOpҐRead_45/DisableCopyOnReadҐRead_45/ReadVariableOpҐRead_46/DisableCopyOnReadҐRead_46/ReadVariableOpҐRead_47/DisableCopyOnReadҐRead_47/ReadVariableOpҐRead_48/DisableCopyOnReadҐRead_48/ReadVariableOpҐRead_49/DisableCopyOnReadҐRead_49/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_50/DisableCopyOnReadҐRead_50/ReadVariableOpҐRead_51/DisableCopyOnReadҐRead_51/ReadVariableOpҐRead_52/DisableCopyOnReadҐRead_52/ReadVariableOpҐRead_53/DisableCopyOnReadҐRead_53/ReadVariableOpҐRead_54/DisableCopyOnReadҐRead_54/ReadVariableOpҐRead_55/DisableCopyOnReadҐRead_55/ReadVariableOpҐRead_56/DisableCopyOnReadҐRead_56/ReadVariableOpҐRead_57/DisableCopyOnReadҐRead_57/ReadVariableOpҐRead_58/DisableCopyOnReadҐRead_58/ReadVariableOpҐRead_59/DisableCopyOnReadҐRead_59/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_60/DisableCopyOnReadҐRead_60/ReadVariableOpҐRead_61/DisableCopyOnReadҐRead_61/ReadVariableOpҐRead_62/DisableCopyOnReadҐRead_62/ReadVariableOpҐRead_63/DisableCopyOnReadҐRead_63/ReadVariableOpҐRead_64/DisableCopyOnReadҐRead_64/ReadVariableOpҐRead_65/DisableCopyOnReadҐRead_65/ReadVariableOpҐRead_66/DisableCopyOnReadҐRead_66/ReadVariableOpҐRead_67/DisableCopyOnReadҐRead_67/ReadVariableOpҐRead_68/DisableCopyOnReadҐRead_68/ReadVariableOpҐRead_69/DisableCopyOnReadҐRead_69/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_70/DisableCopyOnReadҐRead_70/ReadVariableOpҐRead_71/DisableCopyOnReadҐRead_71/ReadVariableOpҐRead_72/DisableCopyOnReadҐRead_72/ReadVariableOpҐRead_73/DisableCopyOnReadҐRead_73/ReadVariableOpҐRead_74/DisableCopyOnReadҐRead_74/ReadVariableOpҐRead_75/DisableCopyOnReadҐRead_75/ReadVariableOpҐRead_76/DisableCopyOnReadҐRead_76/ReadVariableOpҐRead_77/DisableCopyOnReadҐRead_77/ReadVariableOpҐRead_78/DisableCopyOnReadҐRead_78/ReadVariableOpҐRead_79/DisableCopyOnReadҐRead_79/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_80/DisableCopyOnReadҐRead_80/ReadVariableOpҐRead_81/DisableCopyOnReadҐRead_81/ReadVariableOpҐRead_82/DisableCopyOnReadҐRead_82/ReadVariableOpҐRead_83/DisableCopyOnReadҐRead_83/ReadVariableOpҐRead_84/DisableCopyOnReadҐRead_84/ReadVariableOpҐRead_85/DisableCopyOnReadҐRead_85/ReadVariableOpҐRead_86/DisableCopyOnReadҐRead_86/ReadVariableOpҐRead_87/DisableCopyOnReadҐRead_87/ReadVariableOpҐRead_88/DisableCopyOnReadҐRead_88/ReadVariableOpҐRead_89/DisableCopyOnReadҐRead_89/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpҐRead_90/DisableCopyOnReadҐRead_90/ReadVariableOpҐRead_91/DisableCopyOnReadҐRead_91/ReadVariableOpҐRead_92/DisableCopyOnReadҐRead_92/ReadVariableOpҐRead_93/DisableCopyOnReadҐRead_93/ReadVariableOpҐRead_94/DisableCopyOnReadҐRead_94/ReadVariableOpҐRead_95/DisableCopyOnReadҐRead_95/ReadVariableOpҐRead_96/DisableCopyOnReadҐRead_96/ReadVariableOpҐRead_97/DisableCopyOnReadҐRead_97/ReadVariableOpҐRead_98/DisableCopyOnReadҐRead_98/ReadVariableOpҐRead_99/DisableCopyOnReadҐRead_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv2d_kernel"/device:CPU:0*
_output_shapes
 ®
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv2d_bias"/device:CPU:0*
_output_shapes
 †
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ж
Read_2/DisableCopyOnReadDisableCopyOnRead2read_2_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Ѓ
Read_2/ReadVariableOpReadVariableOp2read_2_disablecopyonread_batch_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_3/DisableCopyOnReadDisableCopyOnRead1read_3_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 ≠
Read_3/ReadVariableOpReadVariableOp1read_3_disablecopyonread_batch_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@М
Read_4/DisableCopyOnReadDisableCopyOnRead8read_4_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 і
Read_4/ReadVariableOpReadVariableOp8read_4_disablecopyonread_batch_normalization_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_5/DisableCopyOnReadDisableCopyOnRead<read_5_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 Є
Read_5/ReadVariableOpReadVariableOp<read_5_disablecopyonread_batch_normalization_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ґ
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@И
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 ∞
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_1_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 ѓ
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_1_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_1_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_1_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv2d_2_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv2d_2_bias"/device:CPU:0*
_output_shapes
 •
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv2d_2_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_batch_normalization_2_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 ≤
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_batch_normalization_2_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_2_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_2_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_conv2d_3_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_conv2d_3_bias"/device:CPU:0*
_output_shapes
 •
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_conv2d_3_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_20/DisableCopyOnReadDisableCopyOnRead5read_20_disablecopyonread_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_20/ReadVariableOpReadVariableOp5read_20_disablecopyonread_batch_normalization_3_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_21/DisableCopyOnReadDisableCopyOnRead4read_21_disablecopyonread_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 ≤
Read_21/ReadVariableOpReadVariableOp4read_21_disablecopyonread_batch_normalization_3_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_22/DisableCopyOnReadDisableCopyOnRead;read_22_disablecopyonread_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_22/ReadVariableOpReadVariableOp;read_22_disablecopyonread_batch_normalization_3_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_23/DisableCopyOnReadDisableCopyOnRead?read_23_disablecopyonread_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_23/ReadVariableOpReadVariableOp?read_23_disablecopyonread_batch_normalization_3_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_conv2d_4_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_conv2d_4_bias"/device:CPU:0*
_output_shapes
 •
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_conv2d_4_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_26/DisableCopyOnReadDisableCopyOnRead5read_26_disablecopyonread_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_26/ReadVariableOpReadVariableOp5read_26_disablecopyonread_batch_normalization_4_gamma^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_27/DisableCopyOnReadDisableCopyOnRead4read_27_disablecopyonread_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 ≤
Read_27/ReadVariableOpReadVariableOp4read_27_disablecopyonread_batch_normalization_4_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_28/DisableCopyOnReadDisableCopyOnRead;read_28_disablecopyonread_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_28/ReadVariableOpReadVariableOp;read_28_disablecopyonread_batch_normalization_4_moving_mean^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_29/DisableCopyOnReadDisableCopyOnRead?read_29_disablecopyonread_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_29/ReadVariableOpReadVariableOp?read_29_disablecopyonread_batch_normalization_4_moving_variance^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_30/DisableCopyOnReadDisableCopyOnRead)read_30_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_30/ReadVariableOpReadVariableOp)read_30_disablecopyonread_conv2d_5_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_31/DisableCopyOnReadDisableCopyOnRead'read_31_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 •
Read_31/ReadVariableOpReadVariableOp'read_31_disablecopyonread_conv2d_5_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_32/DisableCopyOnReadDisableCopyOnRead5read_32_disablecopyonread_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_32/ReadVariableOpReadVariableOp5read_32_disablecopyonread_batch_normalization_5_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_33/DisableCopyOnReadDisableCopyOnRead4read_33_disablecopyonread_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 ≤
Read_33/ReadVariableOpReadVariableOp4read_33_disablecopyonread_batch_normalization_5_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_34/DisableCopyOnReadDisableCopyOnRead;read_34_disablecopyonread_batch_normalization_5_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_34/ReadVariableOpReadVariableOp;read_34_disablecopyonread_batch_normalization_5_moving_mean^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_35/DisableCopyOnReadDisableCopyOnRead?read_35_disablecopyonread_batch_normalization_5_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_35/ReadVariableOpReadVariableOp?read_35_disablecopyonread_batch_normalization_5_moving_variance^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_36/DisableCopyOnReadDisableCopyOnRead)read_36_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_36/ReadVariableOpReadVariableOp)read_36_disablecopyonread_conv2d_6_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_37/DisableCopyOnReadDisableCopyOnRead'read_37_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 •
Read_37/ReadVariableOpReadVariableOp'read_37_disablecopyonread_conv2d_6_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_38/DisableCopyOnReadDisableCopyOnRead5read_38_disablecopyonread_batch_normalization_6_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_38/ReadVariableOpReadVariableOp5read_38_disablecopyonread_batch_normalization_6_gamma^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_39/DisableCopyOnReadDisableCopyOnRead4read_39_disablecopyonread_batch_normalization_6_beta"/device:CPU:0*
_output_shapes
 ≤
Read_39/ReadVariableOpReadVariableOp4read_39_disablecopyonread_batch_normalization_6_beta^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_40/DisableCopyOnReadDisableCopyOnRead;read_40_disablecopyonread_batch_normalization_6_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_40/ReadVariableOpReadVariableOp;read_40_disablecopyonread_batch_normalization_6_moving_mean^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_41/DisableCopyOnReadDisableCopyOnRead?read_41_disablecopyonread_batch_normalization_6_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_41/ReadVariableOpReadVariableOp?read_41_disablecopyonread_batch_normalization_6_moving_variance^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_42/DisableCopyOnReadDisableCopyOnRead)read_42_disablecopyonread_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_42/ReadVariableOpReadVariableOp)read_42_disablecopyonread_conv2d_7_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_43/DisableCopyOnReadDisableCopyOnRead'read_43_disablecopyonread_conv2d_7_bias"/device:CPU:0*
_output_shapes
 •
Read_43/ReadVariableOpReadVariableOp'read_43_disablecopyonread_conv2d_7_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_44/DisableCopyOnReadDisableCopyOnRead5read_44_disablecopyonread_batch_normalization_7_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_44/ReadVariableOpReadVariableOp5read_44_disablecopyonread_batch_normalization_7_gamma^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_45/DisableCopyOnReadDisableCopyOnRead4read_45_disablecopyonread_batch_normalization_7_beta"/device:CPU:0*
_output_shapes
 ≤
Read_45/ReadVariableOpReadVariableOp4read_45_disablecopyonread_batch_normalization_7_beta^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_46/DisableCopyOnReadDisableCopyOnRead;read_46_disablecopyonread_batch_normalization_7_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_46/ReadVariableOpReadVariableOp;read_46_disablecopyonread_batch_normalization_7_moving_mean^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_47/DisableCopyOnReadDisableCopyOnRead?read_47_disablecopyonread_batch_normalization_7_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_47/ReadVariableOpReadVariableOp?read_47_disablecopyonread_batch_normalization_7_moving_variance^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_48/DisableCopyOnReadDisableCopyOnRead)read_48_disablecopyonread_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_48/ReadVariableOpReadVariableOp)read_48_disablecopyonread_conv2d_8_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_49/DisableCopyOnReadDisableCopyOnRead'read_49_disablecopyonread_conv2d_8_bias"/device:CPU:0*
_output_shapes
 •
Read_49/ReadVariableOpReadVariableOp'read_49_disablecopyonread_conv2d_8_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_50/DisableCopyOnReadDisableCopyOnRead5read_50_disablecopyonread_batch_normalization_8_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_50/ReadVariableOpReadVariableOp5read_50_disablecopyonread_batch_normalization_8_gamma^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_51/DisableCopyOnReadDisableCopyOnRead4read_51_disablecopyonread_batch_normalization_8_beta"/device:CPU:0*
_output_shapes
 ≤
Read_51/ReadVariableOpReadVariableOp4read_51_disablecopyonread_batch_normalization_8_beta^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_52/DisableCopyOnReadDisableCopyOnRead;read_52_disablecopyonread_batch_normalization_8_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_52/ReadVariableOpReadVariableOp;read_52_disablecopyonread_batch_normalization_8_moving_mean^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_53/DisableCopyOnReadDisableCopyOnRead?read_53_disablecopyonread_batch_normalization_8_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_53/ReadVariableOpReadVariableOp?read_53_disablecopyonread_batch_normalization_8_moving_variance^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_54/DisableCopyOnReadDisableCopyOnRead)read_54_disablecopyonread_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_54/ReadVariableOpReadVariableOp)read_54_disablecopyonread_conv2d_9_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_55/DisableCopyOnReadDisableCopyOnRead'read_55_disablecopyonread_conv2d_9_bias"/device:CPU:0*
_output_shapes
 •
Read_55/ReadVariableOpReadVariableOp'read_55_disablecopyonread_conv2d_9_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_56/DisableCopyOnReadDisableCopyOnRead5read_56_disablecopyonread_batch_normalization_9_gamma"/device:CPU:0*
_output_shapes
 ≥
Read_56/ReadVariableOpReadVariableOp5read_56_disablecopyonread_batch_normalization_9_gamma^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:@Й
Read_57/DisableCopyOnReadDisableCopyOnRead4read_57_disablecopyonread_batch_normalization_9_beta"/device:CPU:0*
_output_shapes
 ≤
Read_57/ReadVariableOpReadVariableOp4read_57_disablecopyonread_batch_normalization_9_beta^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_58/DisableCopyOnReadDisableCopyOnRead;read_58_disablecopyonread_batch_normalization_9_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_58/ReadVariableOpReadVariableOp;read_58_disablecopyonread_batch_normalization_9_moving_mean^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_59/DisableCopyOnReadDisableCopyOnRead?read_59_disablecopyonread_batch_normalization_9_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_59/ReadVariableOpReadVariableOp?read_59_disablecopyonread_batch_normalization_9_moving_variance^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_60/DisableCopyOnReadDisableCopyOnRead*read_60_disablecopyonread_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 і
Read_60/ReadVariableOpReadVariableOp*read_60_disablecopyonread_conv2d_10_kernel^Read_60/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@}
Read_61/DisableCopyOnReadDisableCopyOnRead(read_61_disablecopyonread_conv2d_10_bias"/device:CPU:0*
_output_shapes
 ¶
Read_61/ReadVariableOpReadVariableOp(read_61_disablecopyonread_conv2d_10_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:@Л
Read_62/DisableCopyOnReadDisableCopyOnRead6read_62_disablecopyonread_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 і
Read_62/ReadVariableOpReadVariableOp6read_62_disablecopyonread_batch_normalization_10_gamma^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:@К
Read_63/DisableCopyOnReadDisableCopyOnRead5read_63_disablecopyonread_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 ≥
Read_63/ReadVariableOpReadVariableOp5read_63_disablecopyonread_batch_normalization_10_beta^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_64/DisableCopyOnReadDisableCopyOnRead<read_64_disablecopyonread_batch_normalization_10_moving_mean"/device:CPU:0*
_output_shapes
 Ї
Read_64/ReadVariableOpReadVariableOp<read_64_disablecopyonread_batch_normalization_10_moving_mean^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:@Х
Read_65/DisableCopyOnReadDisableCopyOnRead@read_65_disablecopyonread_batch_normalization_10_moving_variance"/device:CPU:0*
_output_shapes
 Њ
Read_65/ReadVariableOpReadVariableOp@read_65_disablecopyonread_batch_normalization_10_moving_variance^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_66/DisableCopyOnReadDisableCopyOnRead*read_66_disablecopyonread_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 і
Read_66/ReadVariableOpReadVariableOp*read_66_disablecopyonread_conv2d_11_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*&
_output_shapes
:@}
Read_67/DisableCopyOnReadDisableCopyOnRead(read_67_disablecopyonread_conv2d_11_bias"/device:CPU:0*
_output_shapes
 ¶
Read_67/ReadVariableOpReadVariableOp(read_67_disablecopyonread_conv2d_11_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_68/DisableCopyOnReadDisableCopyOnRead#read_68_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_68/ReadVariableOpReadVariableOp#read_68_disablecopyonread_iteration^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	h
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: _
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_69/DisableCopyOnReadDisableCopyOnRead'read_69_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 °
Read_69/ReadVariableOpReadVariableOp'read_69_disablecopyonread_learning_rate^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_70/DisableCopyOnReadDisableCopyOnRead.read_70_disablecopyonread_adam_m_conv2d_kernel"/device:CPU:0*
_output_shapes
 Є
Read_70/ReadVariableOpReadVariableOp.read_70_disablecopyonread_adam_m_conv2d_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Г
Read_71/DisableCopyOnReadDisableCopyOnRead.read_71_disablecopyonread_adam_v_conv2d_kernel"/device:CPU:0*
_output_shapes
 Є
Read_71/ReadVariableOpReadVariableOp.read_71_disablecopyonread_adam_v_conv2d_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Б
Read_72/DisableCopyOnReadDisableCopyOnRead,read_72_disablecopyonread_adam_m_conv2d_bias"/device:CPU:0*
_output_shapes
 ™
Read_72/ReadVariableOpReadVariableOp,read_72_disablecopyonread_adam_m_conv2d_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:@Б
Read_73/DisableCopyOnReadDisableCopyOnRead,read_73_disablecopyonread_adam_v_conv2d_bias"/device:CPU:0*
_output_shapes
 ™
Read_73/ReadVariableOpReadVariableOp,read_73_disablecopyonread_adam_v_conv2d_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:@П
Read_74/DisableCopyOnReadDisableCopyOnRead:read_74_disablecopyonread_adam_m_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Є
Read_74/ReadVariableOpReadVariableOp:read_74_disablecopyonread_adam_m_batch_normalization_gamma^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:@П
Read_75/DisableCopyOnReadDisableCopyOnRead:read_75_disablecopyonread_adam_v_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Є
Read_75/ReadVariableOpReadVariableOp:read_75_disablecopyonread_adam_v_batch_normalization_gamma^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:@О
Read_76/DisableCopyOnReadDisableCopyOnRead9read_76_disablecopyonread_adam_m_batch_normalization_beta"/device:CPU:0*
_output_shapes
 Ј
Read_76/ReadVariableOpReadVariableOp9read_76_disablecopyonread_adam_m_batch_normalization_beta^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:@О
Read_77/DisableCopyOnReadDisableCopyOnRead9read_77_disablecopyonread_adam_v_batch_normalization_beta"/device:CPU:0*
_output_shapes
 Ј
Read_77/ReadVariableOpReadVariableOp9read_77_disablecopyonread_adam_v_batch_normalization_beta^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_78/DisableCopyOnReadDisableCopyOnRead0read_78_disablecopyonread_adam_m_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_78/ReadVariableOpReadVariableOp0read_78_disablecopyonread_adam_m_conv2d_1_kernel^Read_78/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_79/DisableCopyOnReadDisableCopyOnRead0read_79_disablecopyonread_adam_v_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_79/ReadVariableOpReadVariableOp0read_79_disablecopyonread_adam_v_conv2d_1_kernel^Read_79/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_80/DisableCopyOnReadDisableCopyOnRead.read_80_disablecopyonread_adam_m_conv2d_1_bias"/device:CPU:0*
_output_shapes
 ђ
Read_80/ReadVariableOpReadVariableOp.read_80_disablecopyonread_adam_m_conv2d_1_bias^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:@Г
Read_81/DisableCopyOnReadDisableCopyOnRead.read_81_disablecopyonread_adam_v_conv2d_1_bias"/device:CPU:0*
_output_shapes
 ђ
Read_81/ReadVariableOpReadVariableOp.read_81_disablecopyonread_adam_v_conv2d_1_bias^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_82/DisableCopyOnReadDisableCopyOnRead<read_82_disablecopyonread_adam_m_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_82/ReadVariableOpReadVariableOp<read_82_disablecopyonread_adam_m_batch_normalization_1_gamma^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_83/DisableCopyOnReadDisableCopyOnRead<read_83_disablecopyonread_adam_v_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_83/ReadVariableOpReadVariableOp<read_83_disablecopyonread_adam_v_batch_normalization_1_gamma^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_84/DisableCopyOnReadDisableCopyOnRead;read_84_disablecopyonread_adam_m_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 є
Read_84/ReadVariableOpReadVariableOp;read_84_disablecopyonread_adam_m_batch_normalization_1_beta^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_85/DisableCopyOnReadDisableCopyOnRead;read_85_disablecopyonread_adam_v_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 є
Read_85/ReadVariableOpReadVariableOp;read_85_disablecopyonread_adam_v_batch_normalization_1_beta^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_86/DisableCopyOnReadDisableCopyOnRead0read_86_disablecopyonread_adam_m_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_86/ReadVariableOpReadVariableOp0read_86_disablecopyonread_adam_m_conv2d_2_kernel^Read_86/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_87/DisableCopyOnReadDisableCopyOnRead0read_87_disablecopyonread_adam_v_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_87/ReadVariableOpReadVariableOp0read_87_disablecopyonread_adam_v_conv2d_2_kernel^Read_87/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_88/DisableCopyOnReadDisableCopyOnRead.read_88_disablecopyonread_adam_m_conv2d_2_bias"/device:CPU:0*
_output_shapes
 ђ
Read_88/ReadVariableOpReadVariableOp.read_88_disablecopyonread_adam_m_conv2d_2_bias^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:@Г
Read_89/DisableCopyOnReadDisableCopyOnRead.read_89_disablecopyonread_adam_v_conv2d_2_bias"/device:CPU:0*
_output_shapes
 ђ
Read_89/ReadVariableOpReadVariableOp.read_89_disablecopyonread_adam_v_conv2d_2_bias^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_90/DisableCopyOnReadDisableCopyOnRead<read_90_disablecopyonread_adam_m_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_90/ReadVariableOpReadVariableOp<read_90_disablecopyonread_adam_m_batch_normalization_2_gamma^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_91/DisableCopyOnReadDisableCopyOnRead<read_91_disablecopyonread_adam_v_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_91/ReadVariableOpReadVariableOp<read_91_disablecopyonread_adam_v_batch_normalization_2_gamma^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_92/DisableCopyOnReadDisableCopyOnRead;read_92_disablecopyonread_adam_m_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 є
Read_92/ReadVariableOpReadVariableOp;read_92_disablecopyonread_adam_m_batch_normalization_2_beta^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_93/DisableCopyOnReadDisableCopyOnRead;read_93_disablecopyonread_adam_v_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 є
Read_93/ReadVariableOpReadVariableOp;read_93_disablecopyonread_adam_v_batch_normalization_2_beta^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_94/DisableCopyOnReadDisableCopyOnRead0read_94_disablecopyonread_adam_m_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_94/ReadVariableOpReadVariableOp0read_94_disablecopyonread_adam_m_conv2d_3_kernel^Read_94/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_95/DisableCopyOnReadDisableCopyOnRead0read_95_disablecopyonread_adam_v_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_95/ReadVariableOpReadVariableOp0read_95_disablecopyonread_adam_v_conv2d_3_kernel^Read_95/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_96/DisableCopyOnReadDisableCopyOnRead.read_96_disablecopyonread_adam_m_conv2d_3_bias"/device:CPU:0*
_output_shapes
 ђ
Read_96/ReadVariableOpReadVariableOp.read_96_disablecopyonread_adam_m_conv2d_3_bias^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes
:@Г
Read_97/DisableCopyOnReadDisableCopyOnRead.read_97_disablecopyonread_adam_v_conv2d_3_bias"/device:CPU:0*
_output_shapes
 ђ
Read_97/ReadVariableOpReadVariableOp.read_97_disablecopyonread_adam_v_conv2d_3_bias^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_98/DisableCopyOnReadDisableCopyOnRead<read_98_disablecopyonread_adam_m_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_98/ReadVariableOpReadVariableOp<read_98_disablecopyonread_adam_m_batch_normalization_3_gamma^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_99/DisableCopyOnReadDisableCopyOnRead<read_99_disablecopyonread_adam_v_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_99/ReadVariableOpReadVariableOp<read_99_disablecopyonread_adam_v_batch_normalization_3_gamma^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_100/DisableCopyOnReadDisableCopyOnRead<read_100_disablecopyonread_adam_m_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 Љ
Read_100/ReadVariableOpReadVariableOp<read_100_disablecopyonread_adam_m_batch_normalization_3_beta^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_101/DisableCopyOnReadDisableCopyOnRead<read_101_disablecopyonread_adam_v_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 Љ
Read_101/ReadVariableOpReadVariableOp<read_101_disablecopyonread_adam_v_batch_normalization_3_beta^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_102/DisableCopyOnReadDisableCopyOnRead1read_102_disablecopyonread_adam_m_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 љ
Read_102/ReadVariableOpReadVariableOp1read_102_disablecopyonread_adam_m_conv2d_4_kernel^Read_102/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@З
Read_103/DisableCopyOnReadDisableCopyOnRead1read_103_disablecopyonread_adam_v_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 љ
Read_103/ReadVariableOpReadVariableOp1read_103_disablecopyonread_adam_v_conv2d_4_kernel^Read_103/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_104/DisableCopyOnReadDisableCopyOnRead/read_104_disablecopyonread_adam_m_conv2d_4_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_104/ReadVariableOpReadVariableOp/read_104_disablecopyonread_adam_m_conv2d_4_bias^Read_104/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_105/DisableCopyOnReadDisableCopyOnRead/read_105_disablecopyonread_adam_v_conv2d_4_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_105/ReadVariableOpReadVariableOp/read_105_disablecopyonread_adam_v_conv2d_4_bias^Read_105/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_106/DisableCopyOnReadDisableCopyOnRead=read_106_disablecopyonread_adam_m_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 љ
Read_106/ReadVariableOpReadVariableOp=read_106_disablecopyonread_adam_m_batch_normalization_4_gamma^Read_106/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_107/DisableCopyOnReadDisableCopyOnRead=read_107_disablecopyonread_adam_v_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 љ
Read_107/ReadVariableOpReadVariableOp=read_107_disablecopyonread_adam_v_batch_normalization_4_gamma^Read_107/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_108/DisableCopyOnReadDisableCopyOnRead<read_108_disablecopyonread_adam_m_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 Љ
Read_108/ReadVariableOpReadVariableOp<read_108_disablecopyonread_adam_m_batch_normalization_4_beta^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_109/DisableCopyOnReadDisableCopyOnRead<read_109_disablecopyonread_adam_v_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 Љ
Read_109/ReadVariableOpReadVariableOp<read_109_disablecopyonread_adam_v_batch_normalization_4_beta^Read_109/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_110/DisableCopyOnReadDisableCopyOnRead1read_110_disablecopyonread_adam_m_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 љ
Read_110/ReadVariableOpReadVariableOp1read_110_disablecopyonread_adam_m_conv2d_5_kernel^Read_110/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@З
Read_111/DisableCopyOnReadDisableCopyOnRead1read_111_disablecopyonread_adam_v_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 љ
Read_111/ReadVariableOpReadVariableOp1read_111_disablecopyonread_adam_v_conv2d_5_kernel^Read_111/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_112/DisableCopyOnReadDisableCopyOnRead/read_112_disablecopyonread_adam_m_conv2d_5_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_112/ReadVariableOpReadVariableOp/read_112_disablecopyonread_adam_m_conv2d_5_bias^Read_112/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_113/DisableCopyOnReadDisableCopyOnRead/read_113_disablecopyonread_adam_v_conv2d_5_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_113/ReadVariableOpReadVariableOp/read_113_disablecopyonread_adam_v_conv2d_5_bias^Read_113/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_114/DisableCopyOnReadDisableCopyOnRead=read_114_disablecopyonread_adam_m_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 љ
Read_114/ReadVariableOpReadVariableOp=read_114_disablecopyonread_adam_m_batch_normalization_5_gamma^Read_114/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_115/DisableCopyOnReadDisableCopyOnRead=read_115_disablecopyonread_adam_v_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 љ
Read_115/ReadVariableOpReadVariableOp=read_115_disablecopyonread_adam_v_batch_normalization_5_gamma^Read_115/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_116/DisableCopyOnReadDisableCopyOnRead<read_116_disablecopyonread_adam_m_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 Љ
Read_116/ReadVariableOpReadVariableOp<read_116_disablecopyonread_adam_m_batch_normalization_5_beta^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_117/DisableCopyOnReadDisableCopyOnRead<read_117_disablecopyonread_adam_v_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 Љ
Read_117/ReadVariableOpReadVariableOp<read_117_disablecopyonread_adam_v_batch_normalization_5_beta^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_118/DisableCopyOnReadDisableCopyOnRead1read_118_disablecopyonread_adam_m_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 љ
Read_118/ReadVariableOpReadVariableOp1read_118_disablecopyonread_adam_m_conv2d_6_kernel^Read_118/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_236IdentityRead_118/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_237IdentityIdentity_236:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@З
Read_119/DisableCopyOnReadDisableCopyOnRead1read_119_disablecopyonread_adam_v_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 љ
Read_119/ReadVariableOpReadVariableOp1read_119_disablecopyonread_adam_v_conv2d_6_kernel^Read_119/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_238IdentityRead_119/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_239IdentityIdentity_238:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_120/DisableCopyOnReadDisableCopyOnRead/read_120_disablecopyonread_adam_m_conv2d_6_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_120/ReadVariableOpReadVariableOp/read_120_disablecopyonread_adam_m_conv2d_6_bias^Read_120/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_240IdentityRead_120/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_241IdentityIdentity_240:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_121/DisableCopyOnReadDisableCopyOnRead/read_121_disablecopyonread_adam_v_conv2d_6_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_121/ReadVariableOpReadVariableOp/read_121_disablecopyonread_adam_v_conv2d_6_bias^Read_121/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_242IdentityRead_121/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_243IdentityIdentity_242:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_122/DisableCopyOnReadDisableCopyOnRead=read_122_disablecopyonread_adam_m_batch_normalization_6_gamma"/device:CPU:0*
_output_shapes
 љ
Read_122/ReadVariableOpReadVariableOp=read_122_disablecopyonread_adam_m_batch_normalization_6_gamma^Read_122/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_244IdentityRead_122/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_245IdentityIdentity_244:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_123/DisableCopyOnReadDisableCopyOnRead=read_123_disablecopyonread_adam_v_batch_normalization_6_gamma"/device:CPU:0*
_output_shapes
 љ
Read_123/ReadVariableOpReadVariableOp=read_123_disablecopyonread_adam_v_batch_normalization_6_gamma^Read_123/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_246IdentityRead_123/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_247IdentityIdentity_246:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_124/DisableCopyOnReadDisableCopyOnRead<read_124_disablecopyonread_adam_m_batch_normalization_6_beta"/device:CPU:0*
_output_shapes
 Љ
Read_124/ReadVariableOpReadVariableOp<read_124_disablecopyonread_adam_m_batch_normalization_6_beta^Read_124/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_248IdentityRead_124/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_249IdentityIdentity_248:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_125/DisableCopyOnReadDisableCopyOnRead<read_125_disablecopyonread_adam_v_batch_normalization_6_beta"/device:CPU:0*
_output_shapes
 Љ
Read_125/ReadVariableOpReadVariableOp<read_125_disablecopyonread_adam_v_batch_normalization_6_beta^Read_125/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_250IdentityRead_125/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_251IdentityIdentity_250:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_126/DisableCopyOnReadDisableCopyOnRead1read_126_disablecopyonread_adam_m_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 љ
Read_126/ReadVariableOpReadVariableOp1read_126_disablecopyonread_adam_m_conv2d_7_kernel^Read_126/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_252IdentityRead_126/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_253IdentityIdentity_252:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@З
Read_127/DisableCopyOnReadDisableCopyOnRead1read_127_disablecopyonread_adam_v_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 љ
Read_127/ReadVariableOpReadVariableOp1read_127_disablecopyonread_adam_v_conv2d_7_kernel^Read_127/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_254IdentityRead_127/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_255IdentityIdentity_254:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_128/DisableCopyOnReadDisableCopyOnRead/read_128_disablecopyonread_adam_m_conv2d_7_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_128/ReadVariableOpReadVariableOp/read_128_disablecopyonread_adam_m_conv2d_7_bias^Read_128/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_256IdentityRead_128/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_257IdentityIdentity_256:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_129/DisableCopyOnReadDisableCopyOnRead/read_129_disablecopyonread_adam_v_conv2d_7_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_129/ReadVariableOpReadVariableOp/read_129_disablecopyonread_adam_v_conv2d_7_bias^Read_129/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_258IdentityRead_129/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_259IdentityIdentity_258:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_130/DisableCopyOnReadDisableCopyOnRead=read_130_disablecopyonread_adam_m_batch_normalization_7_gamma"/device:CPU:0*
_output_shapes
 љ
Read_130/ReadVariableOpReadVariableOp=read_130_disablecopyonread_adam_m_batch_normalization_7_gamma^Read_130/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_260IdentityRead_130/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_261IdentityIdentity_260:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_131/DisableCopyOnReadDisableCopyOnRead=read_131_disablecopyonread_adam_v_batch_normalization_7_gamma"/device:CPU:0*
_output_shapes
 љ
Read_131/ReadVariableOpReadVariableOp=read_131_disablecopyonread_adam_v_batch_normalization_7_gamma^Read_131/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_262IdentityRead_131/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_263IdentityIdentity_262:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_132/DisableCopyOnReadDisableCopyOnRead<read_132_disablecopyonread_adam_m_batch_normalization_7_beta"/device:CPU:0*
_output_shapes
 Љ
Read_132/ReadVariableOpReadVariableOp<read_132_disablecopyonread_adam_m_batch_normalization_7_beta^Read_132/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_264IdentityRead_132/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_265IdentityIdentity_264:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_133/DisableCopyOnReadDisableCopyOnRead<read_133_disablecopyonread_adam_v_batch_normalization_7_beta"/device:CPU:0*
_output_shapes
 Љ
Read_133/ReadVariableOpReadVariableOp<read_133_disablecopyonread_adam_v_batch_normalization_7_beta^Read_133/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_266IdentityRead_133/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_267IdentityIdentity_266:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_134/DisableCopyOnReadDisableCopyOnRead1read_134_disablecopyonread_adam_m_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 љ
Read_134/ReadVariableOpReadVariableOp1read_134_disablecopyonread_adam_m_conv2d_8_kernel^Read_134/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_268IdentityRead_134/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_269IdentityIdentity_268:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@З
Read_135/DisableCopyOnReadDisableCopyOnRead1read_135_disablecopyonread_adam_v_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 љ
Read_135/ReadVariableOpReadVariableOp1read_135_disablecopyonread_adam_v_conv2d_8_kernel^Read_135/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_270IdentityRead_135/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_271IdentityIdentity_270:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_136/DisableCopyOnReadDisableCopyOnRead/read_136_disablecopyonread_adam_m_conv2d_8_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_136/ReadVariableOpReadVariableOp/read_136_disablecopyonread_adam_m_conv2d_8_bias^Read_136/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_272IdentityRead_136/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_273IdentityIdentity_272:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_137/DisableCopyOnReadDisableCopyOnRead/read_137_disablecopyonread_adam_v_conv2d_8_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_137/ReadVariableOpReadVariableOp/read_137_disablecopyonread_adam_v_conv2d_8_bias^Read_137/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_274IdentityRead_137/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_275IdentityIdentity_274:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_138/DisableCopyOnReadDisableCopyOnRead=read_138_disablecopyonread_adam_m_batch_normalization_8_gamma"/device:CPU:0*
_output_shapes
 љ
Read_138/ReadVariableOpReadVariableOp=read_138_disablecopyonread_adam_m_batch_normalization_8_gamma^Read_138/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_276IdentityRead_138/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_277IdentityIdentity_276:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_139/DisableCopyOnReadDisableCopyOnRead=read_139_disablecopyonread_adam_v_batch_normalization_8_gamma"/device:CPU:0*
_output_shapes
 љ
Read_139/ReadVariableOpReadVariableOp=read_139_disablecopyonread_adam_v_batch_normalization_8_gamma^Read_139/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_278IdentityRead_139/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_279IdentityIdentity_278:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_140/DisableCopyOnReadDisableCopyOnRead<read_140_disablecopyonread_adam_m_batch_normalization_8_beta"/device:CPU:0*
_output_shapes
 Љ
Read_140/ReadVariableOpReadVariableOp<read_140_disablecopyonread_adam_m_batch_normalization_8_beta^Read_140/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_280IdentityRead_140/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_281IdentityIdentity_280:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_141/DisableCopyOnReadDisableCopyOnRead<read_141_disablecopyonread_adam_v_batch_normalization_8_beta"/device:CPU:0*
_output_shapes
 Љ
Read_141/ReadVariableOpReadVariableOp<read_141_disablecopyonread_adam_v_batch_normalization_8_beta^Read_141/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_282IdentityRead_141/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_283IdentityIdentity_282:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_142/DisableCopyOnReadDisableCopyOnRead1read_142_disablecopyonread_adam_m_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 љ
Read_142/ReadVariableOpReadVariableOp1read_142_disablecopyonread_adam_m_conv2d_9_kernel^Read_142/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_284IdentityRead_142/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_285IdentityIdentity_284:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@З
Read_143/DisableCopyOnReadDisableCopyOnRead1read_143_disablecopyonread_adam_v_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 љ
Read_143/ReadVariableOpReadVariableOp1read_143_disablecopyonread_adam_v_conv2d_9_kernel^Read_143/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_286IdentityRead_143/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_287IdentityIdentity_286:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Е
Read_144/DisableCopyOnReadDisableCopyOnRead/read_144_disablecopyonread_adam_m_conv2d_9_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_144/ReadVariableOpReadVariableOp/read_144_disablecopyonread_adam_m_conv2d_9_bias^Read_144/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_288IdentityRead_144/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_289IdentityIdentity_288:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_145/DisableCopyOnReadDisableCopyOnRead/read_145_disablecopyonread_adam_v_conv2d_9_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_145/ReadVariableOpReadVariableOp/read_145_disablecopyonread_adam_v_conv2d_9_bias^Read_145/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_290IdentityRead_145/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_291IdentityIdentity_290:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_146/DisableCopyOnReadDisableCopyOnRead=read_146_disablecopyonread_adam_m_batch_normalization_9_gamma"/device:CPU:0*
_output_shapes
 љ
Read_146/ReadVariableOpReadVariableOp=read_146_disablecopyonread_adam_m_batch_normalization_9_gamma^Read_146/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_292IdentityRead_146/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_293IdentityIdentity_292:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_147/DisableCopyOnReadDisableCopyOnRead=read_147_disablecopyonread_adam_v_batch_normalization_9_gamma"/device:CPU:0*
_output_shapes
 љ
Read_147/ReadVariableOpReadVariableOp=read_147_disablecopyonread_adam_v_batch_normalization_9_gamma^Read_147/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_294IdentityRead_147/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_295IdentityIdentity_294:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_148/DisableCopyOnReadDisableCopyOnRead<read_148_disablecopyonread_adam_m_batch_normalization_9_beta"/device:CPU:0*
_output_shapes
 Љ
Read_148/ReadVariableOpReadVariableOp<read_148_disablecopyonread_adam_m_batch_normalization_9_beta^Read_148/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_296IdentityRead_148/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_297IdentityIdentity_296:output:0"/device:CPU:0*
T0*
_output_shapes
:@Т
Read_149/DisableCopyOnReadDisableCopyOnRead<read_149_disablecopyonread_adam_v_batch_normalization_9_beta"/device:CPU:0*
_output_shapes
 Љ
Read_149/ReadVariableOpReadVariableOp<read_149_disablecopyonread_adam_v_batch_normalization_9_beta^Read_149/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_298IdentityRead_149/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_299IdentityIdentity_298:output:0"/device:CPU:0*
T0*
_output_shapes
:@И
Read_150/DisableCopyOnReadDisableCopyOnRead2read_150_disablecopyonread_adam_m_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_150/ReadVariableOpReadVariableOp2read_150_disablecopyonread_adam_m_conv2d_10_kernel^Read_150/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_300IdentityRead_150/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_301IdentityIdentity_300:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@И
Read_151/DisableCopyOnReadDisableCopyOnRead2read_151_disablecopyonread_adam_v_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_151/ReadVariableOpReadVariableOp2read_151_disablecopyonread_adam_v_conv2d_10_kernel^Read_151/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_302IdentityRead_151/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_303IdentityIdentity_302:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Ж
Read_152/DisableCopyOnReadDisableCopyOnRead0read_152_disablecopyonread_adam_m_conv2d_10_bias"/device:CPU:0*
_output_shapes
 ∞
Read_152/ReadVariableOpReadVariableOp0read_152_disablecopyonread_adam_m_conv2d_10_bias^Read_152/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_304IdentityRead_152/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_305IdentityIdentity_304:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ж
Read_153/DisableCopyOnReadDisableCopyOnRead0read_153_disablecopyonread_adam_v_conv2d_10_bias"/device:CPU:0*
_output_shapes
 ∞
Read_153/ReadVariableOpReadVariableOp0read_153_disablecopyonread_adam_v_conv2d_10_bias^Read_153/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_306IdentityRead_153/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_307IdentityIdentity_306:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_154/DisableCopyOnReadDisableCopyOnRead>read_154_disablecopyonread_adam_m_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 Њ
Read_154/ReadVariableOpReadVariableOp>read_154_disablecopyonread_adam_m_batch_normalization_10_gamma^Read_154/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_308IdentityRead_154/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_309IdentityIdentity_308:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_155/DisableCopyOnReadDisableCopyOnRead>read_155_disablecopyonread_adam_v_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 Њ
Read_155/ReadVariableOpReadVariableOp>read_155_disablecopyonread_adam_v_batch_normalization_10_gamma^Read_155/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_310IdentityRead_155/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_311IdentityIdentity_310:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_156/DisableCopyOnReadDisableCopyOnRead=read_156_disablecopyonread_adam_m_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 љ
Read_156/ReadVariableOpReadVariableOp=read_156_disablecopyonread_adam_m_batch_normalization_10_beta^Read_156/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_312IdentityRead_156/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_313IdentityIdentity_312:output:0"/device:CPU:0*
T0*
_output_shapes
:@У
Read_157/DisableCopyOnReadDisableCopyOnRead=read_157_disablecopyonread_adam_v_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 љ
Read_157/ReadVariableOpReadVariableOp=read_157_disablecopyonread_adam_v_batch_normalization_10_beta^Read_157/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_314IdentityRead_157/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_315IdentityIdentity_314:output:0"/device:CPU:0*
T0*
_output_shapes
:@И
Read_158/DisableCopyOnReadDisableCopyOnRead2read_158_disablecopyonread_adam_m_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_158/ReadVariableOpReadVariableOp2read_158_disablecopyonread_adam_m_conv2d_11_kernel^Read_158/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0y
Identity_316IdentityRead_158/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_317IdentityIdentity_316:output:0"/device:CPU:0*
T0*&
_output_shapes
:@И
Read_159/DisableCopyOnReadDisableCopyOnRead2read_159_disablecopyonread_adam_v_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_159/ReadVariableOpReadVariableOp2read_159_disablecopyonread_adam_v_conv2d_11_kernel^Read_159/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0y
Identity_318IdentityRead_159/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_319IdentityIdentity_318:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Ж
Read_160/DisableCopyOnReadDisableCopyOnRead0read_160_disablecopyonread_adam_m_conv2d_11_bias"/device:CPU:0*
_output_shapes
 ∞
Read_160/ReadVariableOpReadVariableOp0read_160_disablecopyonread_adam_m_conv2d_11_bias^Read_160/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_320IdentityRead_160/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_321IdentityIdentity_320:output:0"/device:CPU:0*
T0*
_output_shapes
:Ж
Read_161/DisableCopyOnReadDisableCopyOnRead0read_161_disablecopyonread_adam_v_conv2d_11_bias"/device:CPU:0*
_output_shapes
 ∞
Read_161/ReadVariableOpReadVariableOp0read_161_disablecopyonread_adam_v_conv2d_11_bias^Read_161/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_322IdentityRead_161/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_323IdentityIdentity_322:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_162/DisableCopyOnReadDisableCopyOnRead read_162_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Ь
Read_162/ReadVariableOpReadVariableOp read_162_disablecopyonread_total^Read_162/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_324IdentityRead_162/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_325IdentityIdentity_324:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_163/DisableCopyOnReadDisableCopyOnRead read_163_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ь
Read_163/ReadVariableOpReadVariableOp read_163_disablecopyonread_count^Read_163/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_326IdentityRead_163/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_327IdentityIdentity_326:output:0"/device:CPU:0*
T0*
_output_shapes
: нF
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:•*
dtype0*ХF
valueЛFBИF•B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЉ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:•*
dtype0*а
value÷B”•B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Е 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0Identity_237:output:0Identity_239:output:0Identity_241:output:0Identity_243:output:0Identity_245:output:0Identity_247:output:0Identity_249:output:0Identity_251:output:0Identity_253:output:0Identity_255:output:0Identity_257:output:0Identity_259:output:0Identity_261:output:0Identity_263:output:0Identity_265:output:0Identity_267:output:0Identity_269:output:0Identity_271:output:0Identity_273:output:0Identity_275:output:0Identity_277:output:0Identity_279:output:0Identity_281:output:0Identity_283:output:0Identity_285:output:0Identity_287:output:0Identity_289:output:0Identity_291:output:0Identity_293:output:0Identity_295:output:0Identity_297:output:0Identity_299:output:0Identity_301:output:0Identity_303:output:0Identity_305:output:0Identity_307:output:0Identity_309:output:0Identity_311:output:0Identity_313:output:0Identity_315:output:0Identity_317:output:0Identity_319:output:0Identity_321:output:0Identity_323:output:0Identity_325:output:0Identity_327:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *ґ
dtypesЂ
®2•	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_328Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_329IdentityIdentity_328:output:0^NoOp*
T0*
_output_shapes
: УE
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_118/DisableCopyOnRead^Read_118/ReadVariableOp^Read_119/DisableCopyOnRead^Read_119/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_120/DisableCopyOnRead^Read_120/ReadVariableOp^Read_121/DisableCopyOnRead^Read_121/ReadVariableOp^Read_122/DisableCopyOnRead^Read_122/ReadVariableOp^Read_123/DisableCopyOnRead^Read_123/ReadVariableOp^Read_124/DisableCopyOnRead^Read_124/ReadVariableOp^Read_125/DisableCopyOnRead^Read_125/ReadVariableOp^Read_126/DisableCopyOnRead^Read_126/ReadVariableOp^Read_127/DisableCopyOnRead^Read_127/ReadVariableOp^Read_128/DisableCopyOnRead^Read_128/ReadVariableOp^Read_129/DisableCopyOnRead^Read_129/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_130/DisableCopyOnRead^Read_130/ReadVariableOp^Read_131/DisableCopyOnRead^Read_131/ReadVariableOp^Read_132/DisableCopyOnRead^Read_132/ReadVariableOp^Read_133/DisableCopyOnRead^Read_133/ReadVariableOp^Read_134/DisableCopyOnRead^Read_134/ReadVariableOp^Read_135/DisableCopyOnRead^Read_135/ReadVariableOp^Read_136/DisableCopyOnRead^Read_136/ReadVariableOp^Read_137/DisableCopyOnRead^Read_137/ReadVariableOp^Read_138/DisableCopyOnRead^Read_138/ReadVariableOp^Read_139/DisableCopyOnRead^Read_139/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_140/DisableCopyOnRead^Read_140/ReadVariableOp^Read_141/DisableCopyOnRead^Read_141/ReadVariableOp^Read_142/DisableCopyOnRead^Read_142/ReadVariableOp^Read_143/DisableCopyOnRead^Read_143/ReadVariableOp^Read_144/DisableCopyOnRead^Read_144/ReadVariableOp^Read_145/DisableCopyOnRead^Read_145/ReadVariableOp^Read_146/DisableCopyOnRead^Read_146/ReadVariableOp^Read_147/DisableCopyOnRead^Read_147/ReadVariableOp^Read_148/DisableCopyOnRead^Read_148/ReadVariableOp^Read_149/DisableCopyOnRead^Read_149/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_150/DisableCopyOnRead^Read_150/ReadVariableOp^Read_151/DisableCopyOnRead^Read_151/ReadVariableOp^Read_152/DisableCopyOnRead^Read_152/ReadVariableOp^Read_153/DisableCopyOnRead^Read_153/ReadVariableOp^Read_154/DisableCopyOnRead^Read_154/ReadVariableOp^Read_155/DisableCopyOnRead^Read_155/ReadVariableOp^Read_156/DisableCopyOnRead^Read_156/ReadVariableOp^Read_157/DisableCopyOnRead^Read_157/ReadVariableOp^Read_158/DisableCopyOnRead^Read_158/ReadVariableOp^Read_159/DisableCopyOnRead^Read_159/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_160/DisableCopyOnRead^Read_160/ReadVariableOp^Read_161/DisableCopyOnRead^Read_161/ReadVariableOp^Read_162/DisableCopyOnRead^Read_162/ReadVariableOp^Read_163/DisableCopyOnRead^Read_163/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_329Identity_329:output:0*(
_construction_contextkEagerRuntime*б
_input_shapesѕ
ћ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp28
Read_118/DisableCopyOnReadRead_118/DisableCopyOnRead22
Read_118/ReadVariableOpRead_118/ReadVariableOp28
Read_119/DisableCopyOnReadRead_119/DisableCopyOnRead22
Read_119/ReadVariableOpRead_119/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp28
Read_120/DisableCopyOnReadRead_120/DisableCopyOnRead22
Read_120/ReadVariableOpRead_120/ReadVariableOp28
Read_121/DisableCopyOnReadRead_121/DisableCopyOnRead22
Read_121/ReadVariableOpRead_121/ReadVariableOp28
Read_122/DisableCopyOnReadRead_122/DisableCopyOnRead22
Read_122/ReadVariableOpRead_122/ReadVariableOp28
Read_123/DisableCopyOnReadRead_123/DisableCopyOnRead22
Read_123/ReadVariableOpRead_123/ReadVariableOp28
Read_124/DisableCopyOnReadRead_124/DisableCopyOnRead22
Read_124/ReadVariableOpRead_124/ReadVariableOp28
Read_125/DisableCopyOnReadRead_125/DisableCopyOnRead22
Read_125/ReadVariableOpRead_125/ReadVariableOp28
Read_126/DisableCopyOnReadRead_126/DisableCopyOnRead22
Read_126/ReadVariableOpRead_126/ReadVariableOp28
Read_127/DisableCopyOnReadRead_127/DisableCopyOnRead22
Read_127/ReadVariableOpRead_127/ReadVariableOp28
Read_128/DisableCopyOnReadRead_128/DisableCopyOnRead22
Read_128/ReadVariableOpRead_128/ReadVariableOp28
Read_129/DisableCopyOnReadRead_129/DisableCopyOnRead22
Read_129/ReadVariableOpRead_129/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp28
Read_130/DisableCopyOnReadRead_130/DisableCopyOnRead22
Read_130/ReadVariableOpRead_130/ReadVariableOp28
Read_131/DisableCopyOnReadRead_131/DisableCopyOnRead22
Read_131/ReadVariableOpRead_131/ReadVariableOp28
Read_132/DisableCopyOnReadRead_132/DisableCopyOnRead22
Read_132/ReadVariableOpRead_132/ReadVariableOp28
Read_133/DisableCopyOnReadRead_133/DisableCopyOnRead22
Read_133/ReadVariableOpRead_133/ReadVariableOp28
Read_134/DisableCopyOnReadRead_134/DisableCopyOnRead22
Read_134/ReadVariableOpRead_134/ReadVariableOp28
Read_135/DisableCopyOnReadRead_135/DisableCopyOnRead22
Read_135/ReadVariableOpRead_135/ReadVariableOp28
Read_136/DisableCopyOnReadRead_136/DisableCopyOnRead22
Read_136/ReadVariableOpRead_136/ReadVariableOp28
Read_137/DisableCopyOnReadRead_137/DisableCopyOnRead22
Read_137/ReadVariableOpRead_137/ReadVariableOp28
Read_138/DisableCopyOnReadRead_138/DisableCopyOnRead22
Read_138/ReadVariableOpRead_138/ReadVariableOp28
Read_139/DisableCopyOnReadRead_139/DisableCopyOnRead22
Read_139/ReadVariableOpRead_139/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp28
Read_140/DisableCopyOnReadRead_140/DisableCopyOnRead22
Read_140/ReadVariableOpRead_140/ReadVariableOp28
Read_141/DisableCopyOnReadRead_141/DisableCopyOnRead22
Read_141/ReadVariableOpRead_141/ReadVariableOp28
Read_142/DisableCopyOnReadRead_142/DisableCopyOnRead22
Read_142/ReadVariableOpRead_142/ReadVariableOp28
Read_143/DisableCopyOnReadRead_143/DisableCopyOnRead22
Read_143/ReadVariableOpRead_143/ReadVariableOp28
Read_144/DisableCopyOnReadRead_144/DisableCopyOnRead22
Read_144/ReadVariableOpRead_144/ReadVariableOp28
Read_145/DisableCopyOnReadRead_145/DisableCopyOnRead22
Read_145/ReadVariableOpRead_145/ReadVariableOp28
Read_146/DisableCopyOnReadRead_146/DisableCopyOnRead22
Read_146/ReadVariableOpRead_146/ReadVariableOp28
Read_147/DisableCopyOnReadRead_147/DisableCopyOnRead22
Read_147/ReadVariableOpRead_147/ReadVariableOp28
Read_148/DisableCopyOnReadRead_148/DisableCopyOnRead22
Read_148/ReadVariableOpRead_148/ReadVariableOp28
Read_149/DisableCopyOnReadRead_149/DisableCopyOnRead22
Read_149/ReadVariableOpRead_149/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp28
Read_150/DisableCopyOnReadRead_150/DisableCopyOnRead22
Read_150/ReadVariableOpRead_150/ReadVariableOp28
Read_151/DisableCopyOnReadRead_151/DisableCopyOnRead22
Read_151/ReadVariableOpRead_151/ReadVariableOp28
Read_152/DisableCopyOnReadRead_152/DisableCopyOnRead22
Read_152/ReadVariableOpRead_152/ReadVariableOp28
Read_153/DisableCopyOnReadRead_153/DisableCopyOnRead22
Read_153/ReadVariableOpRead_153/ReadVariableOp28
Read_154/DisableCopyOnReadRead_154/DisableCopyOnRead22
Read_154/ReadVariableOpRead_154/ReadVariableOp28
Read_155/DisableCopyOnReadRead_155/DisableCopyOnRead22
Read_155/ReadVariableOpRead_155/ReadVariableOp28
Read_156/DisableCopyOnReadRead_156/DisableCopyOnRead22
Read_156/ReadVariableOpRead_156/ReadVariableOp28
Read_157/DisableCopyOnReadRead_157/DisableCopyOnRead22
Read_157/ReadVariableOpRead_157/ReadVariableOp28
Read_158/DisableCopyOnReadRead_158/DisableCopyOnRead22
Read_158/ReadVariableOpRead_158/ReadVariableOp28
Read_159/DisableCopyOnReadRead_159/DisableCopyOnRead22
Read_159/ReadVariableOpRead_159/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp28
Read_160/DisableCopyOnReadRead_160/DisableCopyOnRead22
Read_160/ReadVariableOpRead_160/ReadVariableOp28
Read_161/DisableCopyOnReadRead_161/DisableCopyOnRead22
Read_161/ReadVariableOpRead_161/ReadVariableOp28
Read_162/DisableCopyOnReadRead_162/DisableCopyOnRead22
Read_162/ReadVariableOpRead_162/ReadVariableOp28
Read_163/DisableCopyOnReadRead_163/DisableCopyOnRead22
Read_163/ReadVariableOpRead_163/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_nameconv2d/kernel:+'
%
_user_specified_nameconv2d/bias:95
3
_user_specified_namebatch_normalization/gamma:84
2
_user_specified_namebatch_normalization/beta:?;
9
_user_specified_name!batch_normalization/moving_mean:C?
=
_user_specified_name%#batch_normalization/moving_variance:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv2d_1/bias:;	7
5
_user_specified_namebatch_normalization_1/gamma::
6
4
_user_specified_namebatch_normalization_1/beta:A=
;
_user_specified_name#!batch_normalization_1/moving_mean:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:/+
)
_user_specified_nameconv2d_2/kernel:-)
'
_user_specified_nameconv2d_2/bias:;7
5
_user_specified_namebatch_normalization_2/gamma::6
4
_user_specified_namebatch_normalization_2/beta:A=
;
_user_specified_name#!batch_normalization_2/moving_mean:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_3/bias:;7
5
_user_specified_namebatch_normalization_3/gamma::6
4
_user_specified_namebatch_normalization_3/beta:A=
;
_user_specified_name#!batch_normalization_3/moving_mean:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:;7
5
_user_specified_namebatch_normalization_4/gamma::6
4
_user_specified_namebatch_normalization_4/beta:A=
;
_user_specified_name#!batch_normalization_4/moving_mean:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:/+
)
_user_specified_nameconv2d_5/kernel:- )
'
_user_specified_nameconv2d_5/bias:;!7
5
_user_specified_namebatch_normalization_5/gamma::"6
4
_user_specified_namebatch_normalization_5/beta:A#=
;
_user_specified_name#!batch_normalization_5/moving_mean:E$A
?
_user_specified_name'%batch_normalization_5/moving_variance:/%+
)
_user_specified_nameconv2d_6/kernel:-&)
'
_user_specified_nameconv2d_6/bias:;'7
5
_user_specified_namebatch_normalization_6/gamma::(6
4
_user_specified_namebatch_normalization_6/beta:A)=
;
_user_specified_name#!batch_normalization_6/moving_mean:E*A
?
_user_specified_name'%batch_normalization_6/moving_variance:/++
)
_user_specified_nameconv2d_7/kernel:-,)
'
_user_specified_nameconv2d_7/bias:;-7
5
_user_specified_namebatch_normalization_7/gamma::.6
4
_user_specified_namebatch_normalization_7/beta:A/=
;
_user_specified_name#!batch_normalization_7/moving_mean:E0A
?
_user_specified_name'%batch_normalization_7/moving_variance:/1+
)
_user_specified_nameconv2d_8/kernel:-2)
'
_user_specified_nameconv2d_8/bias:;37
5
_user_specified_namebatch_normalization_8/gamma::46
4
_user_specified_namebatch_normalization_8/beta:A5=
;
_user_specified_name#!batch_normalization_8/moving_mean:E6A
?
_user_specified_name'%batch_normalization_8/moving_variance:/7+
)
_user_specified_nameconv2d_9/kernel:-8)
'
_user_specified_nameconv2d_9/bias:;97
5
_user_specified_namebatch_normalization_9/gamma:::6
4
_user_specified_namebatch_normalization_9/beta:A;=
;
_user_specified_name#!batch_normalization_9/moving_mean:E<A
?
_user_specified_name'%batch_normalization_9/moving_variance:0=,
*
_user_specified_nameconv2d_10/kernel:.>*
(
_user_specified_nameconv2d_10/bias:<?8
6
_user_specified_namebatch_normalization_10/gamma:;@7
5
_user_specified_namebatch_normalization_10/beta:BA>
<
_user_specified_name$"batch_normalization_10/moving_mean:FBB
@
_user_specified_name(&batch_normalization_10/moving_variance:0C,
*
_user_specified_nameconv2d_11/kernel:.D*
(
_user_specified_nameconv2d_11/bias:)E%
#
_user_specified_name	iteration:-F)
'
_user_specified_namelearning_rate:4G0
.
_user_specified_nameAdam/m/conv2d/kernel:4H0
.
_user_specified_nameAdam/v/conv2d/kernel:2I.
,
_user_specified_nameAdam/m/conv2d/bias:2J.
,
_user_specified_nameAdam/v/conv2d/bias:@K<
:
_user_specified_name" Adam/m/batch_normalization/gamma:@L<
:
_user_specified_name" Adam/v/batch_normalization/gamma:?M;
9
_user_specified_name!Adam/m/batch_normalization/beta:?N;
9
_user_specified_name!Adam/v/batch_normalization/beta:6O2
0
_user_specified_nameAdam/m/conv2d_1/kernel:6P2
0
_user_specified_nameAdam/v/conv2d_1/kernel:4Q0
.
_user_specified_nameAdam/m/conv2d_1/bias:4R0
.
_user_specified_nameAdam/v/conv2d_1/bias:BS>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:BT>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:AU=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:AV=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:6W2
0
_user_specified_nameAdam/m/conv2d_2/kernel:6X2
0
_user_specified_nameAdam/v/conv2d_2/kernel:4Y0
.
_user_specified_nameAdam/m/conv2d_2/bias:4Z0
.
_user_specified_nameAdam/v/conv2d_2/bias:B[>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:B\>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:A]=
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:A^=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:6_2
0
_user_specified_nameAdam/m/conv2d_3/kernel:6`2
0
_user_specified_nameAdam/v/conv2d_3/kernel:4a0
.
_user_specified_nameAdam/m/conv2d_3/bias:4b0
.
_user_specified_nameAdam/v/conv2d_3/bias:Bc>
<
_user_specified_name$"Adam/m/batch_normalization_3/gamma:Bd>
<
_user_specified_name$"Adam/v/batch_normalization_3/gamma:Ae=
;
_user_specified_name#!Adam/m/batch_normalization_3/beta:Af=
;
_user_specified_name#!Adam/v/batch_normalization_3/beta:6g2
0
_user_specified_nameAdam/m/conv2d_4/kernel:6h2
0
_user_specified_nameAdam/v/conv2d_4/kernel:4i0
.
_user_specified_nameAdam/m/conv2d_4/bias:4j0
.
_user_specified_nameAdam/v/conv2d_4/bias:Bk>
<
_user_specified_name$"Adam/m/batch_normalization_4/gamma:Bl>
<
_user_specified_name$"Adam/v/batch_normalization_4/gamma:Am=
;
_user_specified_name#!Adam/m/batch_normalization_4/beta:An=
;
_user_specified_name#!Adam/v/batch_normalization_4/beta:6o2
0
_user_specified_nameAdam/m/conv2d_5/kernel:6p2
0
_user_specified_nameAdam/v/conv2d_5/kernel:4q0
.
_user_specified_nameAdam/m/conv2d_5/bias:4r0
.
_user_specified_nameAdam/v/conv2d_5/bias:Bs>
<
_user_specified_name$"Adam/m/batch_normalization_5/gamma:Bt>
<
_user_specified_name$"Adam/v/batch_normalization_5/gamma:Au=
;
_user_specified_name#!Adam/m/batch_normalization_5/beta:Av=
;
_user_specified_name#!Adam/v/batch_normalization_5/beta:6w2
0
_user_specified_nameAdam/m/conv2d_6/kernel:6x2
0
_user_specified_nameAdam/v/conv2d_6/kernel:4y0
.
_user_specified_nameAdam/m/conv2d_6/bias:4z0
.
_user_specified_nameAdam/v/conv2d_6/bias:B{>
<
_user_specified_name$"Adam/m/batch_normalization_6/gamma:B|>
<
_user_specified_name$"Adam/v/batch_normalization_6/gamma:A}=
;
_user_specified_name#!Adam/m/batch_normalization_6/beta:A~=
;
_user_specified_name#!Adam/v/batch_normalization_6/beta:62
0
_user_specified_nameAdam/m/conv2d_7/kernel:7А2
0
_user_specified_nameAdam/v/conv2d_7/kernel:5Б0
.
_user_specified_nameAdam/m/conv2d_7/bias:5В0
.
_user_specified_nameAdam/v/conv2d_7/bias:CГ>
<
_user_specified_name$"Adam/m/batch_normalization_7/gamma:CД>
<
_user_specified_name$"Adam/v/batch_normalization_7/gamma:BЕ=
;
_user_specified_name#!Adam/m/batch_normalization_7/beta:BЖ=
;
_user_specified_name#!Adam/v/batch_normalization_7/beta:7З2
0
_user_specified_nameAdam/m/conv2d_8/kernel:7И2
0
_user_specified_nameAdam/v/conv2d_8/kernel:5Й0
.
_user_specified_nameAdam/m/conv2d_8/bias:5К0
.
_user_specified_nameAdam/v/conv2d_8/bias:CЛ>
<
_user_specified_name$"Adam/m/batch_normalization_8/gamma:CМ>
<
_user_specified_name$"Adam/v/batch_normalization_8/gamma:BН=
;
_user_specified_name#!Adam/m/batch_normalization_8/beta:BО=
;
_user_specified_name#!Adam/v/batch_normalization_8/beta:7П2
0
_user_specified_nameAdam/m/conv2d_9/kernel:7Р2
0
_user_specified_nameAdam/v/conv2d_9/kernel:5С0
.
_user_specified_nameAdam/m/conv2d_9/bias:5Т0
.
_user_specified_nameAdam/v/conv2d_9/bias:CУ>
<
_user_specified_name$"Adam/m/batch_normalization_9/gamma:CФ>
<
_user_specified_name$"Adam/v/batch_normalization_9/gamma:BХ=
;
_user_specified_name#!Adam/m/batch_normalization_9/beta:BЦ=
;
_user_specified_name#!Adam/v/batch_normalization_9/beta:8Ч3
1
_user_specified_nameAdam/m/conv2d_10/kernel:8Ш3
1
_user_specified_nameAdam/v/conv2d_10/kernel:6Щ1
/
_user_specified_nameAdam/m/conv2d_10/bias:6Ъ1
/
_user_specified_nameAdam/v/conv2d_10/bias:DЫ?
=
_user_specified_name%#Adam/m/batch_normalization_10/gamma:DЬ?
=
_user_specified_name%#Adam/v/batch_normalization_10/gamma:CЭ>
<
_user_specified_name$"Adam/m/batch_normalization_10/beta:CЮ>
<
_user_specified_name$"Adam/v/batch_normalization_10/beta:8Я3
1
_user_specified_nameAdam/m/conv2d_11/kernel:8†3
1
_user_specified_nameAdam/v/conv2d_11/kernel:6°1
/
_user_specified_nameAdam/m/conv2d_11/bias:6Ґ1
/
_user_specified_nameAdam/v/conv2d_11/bias:&£!

_user_specified_nametotal:&§!

_user_specified_namecount:>•9

_output_shapes
: 

_user_specified_nameConst
Й
њ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20331

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Д
C
'__inference_re_lu_3_layer_call_fn_19990

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_18396z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Љ«
”
@__inference_model_layer_call_and_return_conditional_losses_18631
input_1&
conv2d_18288:@
conv2d_18290:@'
batch_normalization_18293:@'
batch_normalization_18295:@'
batch_normalization_18297:@'
batch_normalization_18299:@(
conv2d_1_18318:@@
conv2d_1_18320:@)
batch_normalization_1_18323:@)
batch_normalization_1_18325:@)
batch_normalization_1_18327:@)
batch_normalization_1_18329:@(
conv2d_2_18348:@@
conv2d_2_18350:@)
batch_normalization_2_18353:@)
batch_normalization_2_18355:@)
batch_normalization_2_18357:@)
batch_normalization_2_18359:@(
conv2d_3_18378:@@
conv2d_3_18380:@)
batch_normalization_3_18383:@)
batch_normalization_3_18385:@)
batch_normalization_3_18387:@)
batch_normalization_3_18389:@(
conv2d_4_18408:@@
conv2d_4_18410:@)
batch_normalization_4_18413:@)
batch_normalization_4_18415:@)
batch_normalization_4_18417:@)
batch_normalization_4_18419:@(
conv2d_5_18438:@@
conv2d_5_18440:@)
batch_normalization_5_18443:@)
batch_normalization_5_18445:@)
batch_normalization_5_18447:@)
batch_normalization_5_18449:@(
conv2d_6_18468:@@
conv2d_6_18470:@)
batch_normalization_6_18473:@)
batch_normalization_6_18475:@)
batch_normalization_6_18477:@)
batch_normalization_6_18479:@(
conv2d_7_18498:@@
conv2d_7_18500:@)
batch_normalization_7_18503:@)
batch_normalization_7_18505:@)
batch_normalization_7_18507:@)
batch_normalization_7_18509:@(
conv2d_8_18528:@@
conv2d_8_18530:@)
batch_normalization_8_18533:@)
batch_normalization_8_18535:@)
batch_normalization_8_18537:@)
batch_normalization_8_18539:@(
conv2d_9_18558:@@
conv2d_9_18560:@)
batch_normalization_9_18563:@)
batch_normalization_9_18565:@)
batch_normalization_9_18567:@)
batch_normalization_9_18569:@)
conv2d_10_18588:@@
conv2d_10_18590:@*
batch_normalization_10_18593:@*
batch_normalization_10_18595:@*
batch_normalization_10_18597:@*
batch_normalization_10_18599:@)
conv2d_11_18618:@
conv2d_11_18620:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ.batch_normalization_10/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ-batch_normalization_8/StatefulPartitionedCallҐ-batch_normalization_9/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ!conv2d_10/StatefulPartitionedCallҐ!conv2d_11/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallҐ conv2d_9/StatefulPartitionedCallГ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_18288conv2d_18290*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_18287П
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_18293batch_normalization_18295batch_normalization_18297batch_normalization_18299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17611ь
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_18306Ґ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_18318conv2d_1_18320*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18317Э
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_18323batch_normalization_1_18325batch_normalization_1_18327batch_normalization_1_18329*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17673В
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_18336§
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_18348conv2d_2_18350*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_18347Э
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_18353batch_normalization_2_18355batch_normalization_2_18357batch_normalization_2_18359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_17735В
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_18366§
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_18378conv2d_3_18380*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_18377Э
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_18383batch_normalization_3_18385batch_normalization_3_18387batch_normalization_3_18389*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_17797В
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_18396§
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_18408conv2d_4_18410*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_18407Э
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_18413batch_normalization_4_18415batch_normalization_4_18417batch_normalization_4_18419*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17859В
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_18426§
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_5_18438conv2d_5_18440*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_18437Э
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_18443batch_normalization_5_18445batch_normalization_5_18447batch_normalization_5_18449*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_17921В
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_18456§
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_6_18468conv2d_6_18470*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_18467Э
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_18473batch_normalization_6_18475batch_normalization_6_18477batch_normalization_6_18479*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_17983В
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_18486§
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0conv2d_7_18498conv2d_7_18500*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_18497Э
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_18503batch_normalization_7_18505batch_normalization_7_18507batch_normalization_7_18509*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_18045В
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_18516§
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv2d_8_18528conv2d_8_18530*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_18527Э
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_18533batch_normalization_8_18535batch_normalization_8_18537batch_normalization_8_18539*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_18107В
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_18546§
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv2d_9_18558conv2d_9_18560*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_18557Э
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_18563batch_normalization_9_18565batch_normalization_9_18567batch_normalization_9_18569*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18169В
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_18576®
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0conv2d_10_18588conv2d_10_18590*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_18587§
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_18593batch_normalization_10_18595batch_normalization_10_18597batch_normalization_10_18599*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_18231Е
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_18606©
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0conv2d_11_18618conv2d_11_18620*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_18617ш
add/PartitionedCallPartitionedCallinput_1*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_18628Е
IdentityIdentityadd/PartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€’
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapesЄ
µ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:%!

_user_specified_name18288:%!

_user_specified_name18290:%!

_user_specified_name18293:%!

_user_specified_name18295:%!

_user_specified_name18297:%!

_user_specified_name18299:%!

_user_specified_name18318:%!

_user_specified_name18320:%	!

_user_specified_name18323:%
!

_user_specified_name18325:%!

_user_specified_name18327:%!

_user_specified_name18329:%!

_user_specified_name18348:%!

_user_specified_name18350:%!

_user_specified_name18353:%!

_user_specified_name18355:%!

_user_specified_name18357:%!

_user_specified_name18359:%!

_user_specified_name18378:%!

_user_specified_name18380:%!

_user_specified_name18383:%!

_user_specified_name18385:%!

_user_specified_name18387:%!

_user_specified_name18389:%!

_user_specified_name18408:%!

_user_specified_name18410:%!

_user_specified_name18413:%!

_user_specified_name18415:%!

_user_specified_name18417:%!

_user_specified_name18419:%!

_user_specified_name18438:% !

_user_specified_name18440:%!!

_user_specified_name18443:%"!

_user_specified_name18445:%#!

_user_specified_name18447:%$!

_user_specified_name18449:%%!

_user_specified_name18468:%&!

_user_specified_name18470:%'!

_user_specified_name18473:%(!

_user_specified_name18475:%)!

_user_specified_name18477:%*!

_user_specified_name18479:%+!

_user_specified_name18498:%,!

_user_specified_name18500:%-!

_user_specified_name18503:%.!

_user_specified_name18505:%/!

_user_specified_name18507:%0!

_user_specified_name18509:%1!

_user_specified_name18528:%2!

_user_specified_name18530:%3!

_user_specified_name18533:%4!

_user_specified_name18535:%5!

_user_specified_name18537:%6!

_user_specified_name18539:%7!

_user_specified_name18558:%8!

_user_specified_name18560:%9!

_user_specified_name18563:%:!

_user_specified_name18565:%;!

_user_specified_name18567:%<!

_user_specified_name18569:%=!

_user_specified_name18588:%>!

_user_specified_name18590:%?!

_user_specified_name18593:%@!

_user_specified_name18595:%A!

_user_specified_name18597:%B!

_user_specified_name18599:%C!

_user_specified_name18618:%D!

_user_specified_name18620
Ѓ
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_19904

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Д

ќ
3__inference_batch_normalization_layer_call_fn_19663

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17611Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name19653:%!

_user_specified_name19655:%!

_user_specified_name19657:%!

_user_specified_name19659
Й
њ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17859

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_18426

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Д
C
'__inference_re_lu_2_layer_call_fn_19899

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_18366z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ѓ
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_20086

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И

–
5__inference_batch_normalization_6_layer_call_fn_20209

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_17983Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:%!

_user_specified_name20199:%!

_user_specified_name20201:%!

_user_specified_name20203:%!

_user_specified_name20205" L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Џ
serving_default∆
U
input_1J
serving_default_input_1:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€Q
addJ
StatefulPartitionedCall:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€tensorflow/serving/predict:їК
і

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!layer_with_weights-21
!layer-32
"layer-33
#layer_with_weights-22
#layer-34
$layer-35
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_default_save_signature
,	optimizer
-
signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ё
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op"
_tf_keras_layer
к
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance"
_tf_keras_layer
•
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
 P_jit_compiled_convolution_op"
_tf_keras_layer
к
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance"
_tf_keras_layer
•
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op"
_tf_keras_layer
к
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance"
_tf_keras_layer
•
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
в
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias
!Д_jit_compiled_convolution_op"
_tf_keras_layer
х
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
	Лaxis

Мgamma
	Нbeta
Оmoving_mean
Пmoving_variance"
_tf_keras_layer
Ђ
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses
Ьkernel
	Эbias
!Ю_jit_compiled_convolution_op"
_tf_keras_layer
х
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
£__call__
+§&call_and_return_all_conditional_losses
	•axis

¶gamma
	Іbeta
®moving_mean
©moving_variance"
_tf_keras_layer
Ђ
™	variables
Ђtrainable_variables
ђregularization_losses
≠	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
∞	variables
±trainable_variables
≤regularization_losses
≥	keras_api
і__call__
+µ&call_and_return_all_conditional_losses
ґkernel
	Јbias
!Є_jit_compiled_convolution_op"
_tf_keras_layer
х
є	variables
Їtrainable_variables
їregularization_losses
Љ	keras_api
љ__call__
+Њ&call_and_return_all_conditional_losses
	њaxis

јgamma
	Ѕbeta
¬moving_mean
√moving_variance"
_tf_keras_layer
Ђ
ƒ	variables
≈trainable_variables
∆regularization_losses
«	keras_api
»__call__
+…&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses
–kernel
	—bias
!“_jit_compiled_convolution_op"
_tf_keras_layer
х
”	variables
‘trainable_variables
’regularization_losses
÷	keras_api
„__call__
+Ў&call_and_return_all_conditional_losses
	ўaxis

Џgamma
	џbeta
№moving_mean
Ёmoving_variance"
_tf_keras_layer
Ђ
ё	variables
яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses
кkernel
	лbias
!м_jit_compiled_convolution_op"
_tf_keras_layer
х
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses
	уaxis

фgamma
	хbeta
цmoving_mean
чmoving_variance"
_tf_keras_layer
Ђ
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
ю	variables
€trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
Дkernel
	Еbias
!Ж_jit_compiled_convolution_op"
_tf_keras_layer
х
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
	Нaxis

Оgamma
	Пbeta
Рmoving_mean
Сmoving_variance"
_tf_keras_layer
Ђ
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Юkernel
	Яbias
!†_jit_compiled_convolution_op"
_tf_keras_layer
х
°	variables
Ґtrainable_variables
£regularization_losses
§	keras_api
•__call__
+¶&call_and_return_all_conditional_losses
	Іaxis

®gamma
	©beta
™moving_mean
Ђmoving_variance"
_tf_keras_layer
Ђ
ђ	variables
≠trainable_variables
Ѓregularization_losses
ѓ	keras_api
∞__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses
Єkernel
	єbias
!Ї_jit_compiled_convolution_op"
_tf_keras_layer
х
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses
	Ѕaxis

¬gamma
	√beta
ƒmoving_mean
≈moving_variance"
_tf_keras_layer
Ђ
∆	variables
«trainable_variables
»regularization_losses
…	keras_api
 __call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
ћ	variables
Ќtrainable_variables
ќregularization_losses
ѕ	keras_api
–__call__
+—&call_and_return_all_conditional_losses
“kernel
	”bias
!‘_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
и
40
51
>2
?3
@4
A5
N6
O7
X8
Y9
Z10
[11
h12
i13
r14
s15
t16
u17
В18
Г19
М20
Н21
О22
П23
Ь24
Э25
¶26
І27
®28
©29
ґ30
Ј31
ј32
Ѕ33
¬34
√35
–36
—37
Џ38
џ39
№40
Ё41
к42
л43
ф44
х45
ц46
ч47
Д48
Е49
О50
П51
Р52
С53
Ю54
Я55
®56
©57
™58
Ђ59
Є60
є61
¬62
√63
ƒ64
≈65
“66
”67"
trackable_list_wrapper
®
40
51
>2
?3
N4
O5
X6
Y7
h8
i9
r10
s11
В12
Г13
М14
Н15
Ь16
Э17
¶18
І19
ґ20
Ј21
ј22
Ѕ23
–24
—25
Џ26
џ27
к28
л29
ф30
х31
Д32
Е33
О34
П35
Ю36
Я37
®38
©39
Є40
є41
¬42
√43
“44
”45"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
+_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Ѕ
аtrace_0
бtrace_12Ж
%__inference_model_layer_call_fn_18947
%__inference_model_layer_call_fn_19088µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zаtrace_0zбtrace_1
ч
вtrace_0
гtrace_12Љ
@__inference_model_layer_call_and_return_conditional_losses_18631
@__inference_model_layer_call_and_return_conditional_losses_18806µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zвtrace_0zгtrace_1
ЋB»
 __inference__wrapped_model_17593input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
£
д
_variables
е_iterations
ж_learning_rate
з_index_dict
и
_momentums
й_velocities
к_update_step_xla"
experimentalOptimizer
-
лserving_default"
signature_map
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
в
сtrace_02√
&__inference_conv2d_layer_call_fn_19640Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zсtrace_0
э
тtrace_02ё
A__inference_conv2d_layer_call_and_return_conditional_losses_19650Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zтtrace_0
':%@2conv2d/kernel
:@2conv2d/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
Ё
шtrace_0
щtrace_12Ґ
3__inference_batch_normalization_layer_call_fn_19663
3__inference_batch_normalization_layer_call_fn_19676µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zшtrace_0zщtrace_1
У
ъtrace_0
ыtrace_12Ў
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19694
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19712µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zъtrace_0zыtrace_1
 "
trackable_list_wrapper
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ьnon_trainable_variables
эlayers
юmetrics
 €layer_regularization_losses
Аlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
б
Бtrace_02¬
%__inference_re_lu_layer_call_fn_19717Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zБtrace_0
ь
Вtrace_02Ё
@__inference_re_lu_layer_call_and_return_conditional_losses_19722Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zВtrace_0
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
д
Иtrace_02≈
(__inference_conv2d_1_layer_call_fn_19731Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zИtrace_0
€
Йtrace_02а
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19741Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЙtrace_0
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
X0
Y1
Z2
[3"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
б
Пtrace_0
Рtrace_12¶
5__inference_batch_normalization_1_layer_call_fn_19754
5__inference_batch_normalization_1_layer_call_fn_19767µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zПtrace_0zРtrace_1
Ч
Сtrace_0
Тtrace_12№
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19785
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19803µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zСtrace_0zТtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
г
Шtrace_02ƒ
'__inference_re_lu_1_layer_call_fn_19808Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zШtrace_0
ю
Щtrace_02я
B__inference_re_lu_1_layer_call_and_return_conditional_losses_19813Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЩtrace_0
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
д
Яtrace_02≈
(__inference_conv2d_2_layer_call_fn_19822Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЯtrace_0
€
†trace_02а
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19832Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z†trace_0
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
r0
s1
t2
u3"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
°non_trainable_variables
Ґlayers
£metrics
 §layer_regularization_losses
•layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
б
¶trace_0
Іtrace_12¶
5__inference_batch_normalization_2_layer_call_fn_19845
5__inference_batch_normalization_2_layer_call_fn_19858µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z¶trace_0zІtrace_1
Ч
®trace_0
©trace_12№
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19876
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19894µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z®trace_0z©trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
г
ѓtrace_02ƒ
'__inference_re_lu_2_layer_call_fn_19899Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѓtrace_0
ю
∞trace_02я
B__inference_re_lu_2_layer_call_and_return_conditional_losses_19904Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∞trace_0
0
В0
Г1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±non_trainable_variables
≤layers
≥metrics
 іlayer_regularization_losses
µlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
д
ґtrace_02≈
(__inference_conv2d_3_layer_call_fn_19913Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zґtrace_0
€
Јtrace_02а
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19923Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈtrace_0
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
М0
Н1
О2
П3"
trackable_list_wrapper
0
М0
Н1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
б
љtrace_0
Њtrace_12¶
5__inference_batch_normalization_3_layer_call_fn_19936
5__inference_batch_normalization_3_layer_call_fn_19949µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zљtrace_0zЊtrace_1
Ч
њtrace_0
јtrace_12№
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19967
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19985µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zњtrace_0zјtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ѕnon_trainable_variables
¬layers
√metrics
 ƒlayer_regularization_losses
≈layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
г
∆trace_02ƒ
'__inference_re_lu_3_layer_call_fn_19990Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∆trace_0
ю
«trace_02я
B__inference_re_lu_3_layer_call_and_return_conditional_losses_19995Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0
0
Ь0
Э1"
trackable_list_wrapper
0
Ь0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
д
Ќtrace_02≈
(__inference_conv2d_4_layer_call_fn_20004Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЌtrace_0
€
ќtrace_02а
C__inference_conv2d_4_layer_call_and_return_conditional_losses_20014Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zќtrace_0
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
¶0
І1
®2
©3"
trackable_list_wrapper
0
¶0
І1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
Я	variables
†trainable_variables
°regularization_losses
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
б
‘trace_0
’trace_12¶
5__inference_batch_normalization_4_layer_call_fn_20027
5__inference_batch_normalization_4_layer_call_fn_20040µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0z’trace_1
Ч
÷trace_0
„trace_12№
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20058
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20076µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0z„trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
™	variables
Ђtrainable_variables
ђregularization_losses
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
г
Ёtrace_02ƒ
'__inference_re_lu_4_layer_call_fn_20081Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЁtrace_0
ю
ёtrace_02я
B__inference_re_lu_4_layer_call_and_return_conditional_losses_20086Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zёtrace_0
0
ґ0
Ј1"
trackable_list_wrapper
0
ґ0
Ј1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
∞	variables
±trainable_variables
≤regularization_losses
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
д
дtrace_02≈
(__inference_conv2d_5_layer_call_fn_20095Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zдtrace_0
€
еtrace_02а
C__inference_conv2d_5_layer_call_and_return_conditional_losses_20105Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zеtrace_0
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
ј0
Ѕ1
¬2
√3"
trackable_list_wrapper
0
ј0
Ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
є	variables
Їtrainable_variables
їregularization_losses
љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
б
лtrace_0
мtrace_12¶
5__inference_batch_normalization_5_layer_call_fn_20118
5__inference_batch_normalization_5_layer_call_fn_20131µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zлtrace_0zмtrace_1
Ч
нtrace_0
оtrace_12№
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20149
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20167µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zнtrace_0zоtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
ƒ	variables
≈trainable_variables
∆regularization_losses
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
г
фtrace_02ƒ
'__inference_re_lu_5_layer_call_fn_20172Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zфtrace_0
ю
хtrace_02я
B__inference_re_lu_5_layer_call_and_return_conditional_losses_20177Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zхtrace_0
0
–0
—1"
trackable_list_wrapper
0
–0
—1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
 	variables
Ћtrainable_variables
ћregularization_losses
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
д
ыtrace_02≈
(__inference_conv2d_6_layer_call_fn_20186Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zыtrace_0
€
ьtrace_02а
C__inference_conv2d_6_layer_call_and_return_conditional_losses_20196Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zьtrace_0
):'@@2conv2d_6/kernel
:@2conv2d_6/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
Џ0
џ1
№2
Ё3"
trackable_list_wrapper
0
Џ0
џ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
”	variables
‘trainable_variables
’regularization_losses
„__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
б
Вtrace_0
Гtrace_12¶
5__inference_batch_normalization_6_layer_call_fn_20209
5__inference_batch_normalization_6_layer_call_fn_20222µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zВtrace_0zГtrace_1
Ч
Дtrace_0
Еtrace_12№
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20240
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20258µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zДtrace_0zЕtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_6/gamma
(:&@2batch_normalization_6/beta
1:/@ (2!batch_normalization_6/moving_mean
5:3@ (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
ё	variables
яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
г
Лtrace_02ƒ
'__inference_re_lu_6_layer_call_fn_20263Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЛtrace_0
ю
Мtrace_02я
B__inference_re_lu_6_layer_call_and_return_conditional_losses_20268Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zМtrace_0
0
к0
л1"
trackable_list_wrapper
0
к0
л1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
д
Тtrace_02≈
(__inference_conv2d_7_layer_call_fn_20277Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zТtrace_0
€
Уtrace_02а
C__inference_conv2d_7_layer_call_and_return_conditional_losses_20287Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zУtrace_0
):'@@2conv2d_7/kernel
:@2conv2d_7/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
ф0
х1
ц2
ч3"
trackable_list_wrapper
0
ф0
х1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
б
Щtrace_0
Ъtrace_12¶
5__inference_batch_normalization_7_layer_call_fn_20300
5__inference_batch_normalization_7_layer_call_fn_20313µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЩtrace_0zЪtrace_1
Ч
Ыtrace_0
Ьtrace_12№
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20331
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20349µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЫtrace_0zЬtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
г
Ґtrace_02ƒ
'__inference_re_lu_7_layer_call_fn_20354Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zҐtrace_0
ю
£trace_02я
B__inference_re_lu_7_layer_call_and_return_conditional_losses_20359Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z£trace_0
0
Д0
Е1"
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
ю	variables
€trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
д
©trace_02≈
(__inference_conv2d_8_layer_call_fn_20368Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z©trace_0
€
™trace_02а
C__inference_conv2d_8_layer_call_and_return_conditional_losses_20378Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
):'@@2conv2d_8/kernel
:@2conv2d_8/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
О0
П1
Р2
С3"
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
б
∞trace_0
±trace_12¶
5__inference_batch_normalization_8_layer_call_fn_20391
5__inference_batch_normalization_8_layer_call_fn_20404µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∞trace_0z±trace_1
Ч
≤trace_0
≥trace_12№
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20422
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20440µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≤trace_0z≥trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
г
єtrace_02ƒ
'__inference_re_lu_8_layer_call_fn_20445Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zєtrace_0
ю
Їtrace_02я
B__inference_re_lu_8_layer_call_and_return_conditional_losses_20450Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЇtrace_0
0
Ю0
Я1"
trackable_list_wrapper
0
Ю0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
д
јtrace_02≈
(__inference_conv2d_9_layer_call_fn_20459Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zјtrace_0
€
Ѕtrace_02а
C__inference_conv2d_9_layer_call_and_return_conditional_losses_20469Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0
):'@@2conv2d_9/kernel
:@2conv2d_9/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
®0
©1
™2
Ђ3"
trackable_list_wrapper
0
®0
©1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
°	variables
Ґtrainable_variables
£regularization_losses
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
б
«trace_0
»trace_12¶
5__inference_batch_normalization_9_layer_call_fn_20482
5__inference_batch_normalization_9_layer_call_fn_20495µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0z»trace_1
Ч
…trace_0
 trace_12№
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20513
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20531µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z…trace_0z trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_9/gamma
(:&@2batch_normalization_9/beta
1:/@ (2!batch_normalization_9/moving_mean
5:3@ (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ћnon_trainable_variables
ћlayers
Ќmetrics
 ќlayer_regularization_losses
ѕlayer_metrics
ђ	variables
≠trainable_variables
Ѓregularization_losses
∞__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
г
–trace_02ƒ
'__inference_re_lu_9_layer_call_fn_20536Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z–trace_0
ю
—trace_02я
B__inference_re_lu_9_layer_call_and_return_conditional_losses_20541Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z—trace_0
0
Є0
є1"
trackable_list_wrapper
0
Є0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
≤	variables
≥trainable_variables
іregularization_losses
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
е
„trace_02∆
)__inference_conv2d_10_layer_call_fn_20550Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z„trace_0
А
Ўtrace_02б
D__inference_conv2d_10_layer_call_and_return_conditional_losses_20560Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЎtrace_0
*:(@@2conv2d_10/kernel
:@2conv2d_10/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
@
¬0
√1
ƒ2
≈3"
trackable_list_wrapper
0
¬0
√1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ўnon_trainable_variables
Џlayers
џmetrics
 №layer_regularization_losses
Ёlayer_metrics
ї	variables
Љtrainable_variables
љregularization_losses
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
г
ёtrace_0
яtrace_12®
6__inference_batch_normalization_10_layer_call_fn_20573
6__inference_batch_normalization_10_layer_call_fn_20586µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zёtrace_0zяtrace_1
Щ
аtrace_0
бtrace_12ё
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20604
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20622µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zаtrace_0zбtrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_10/gamma
):'@2batch_normalization_10/beta
2:0@ (2"batch_normalization_10/moving_mean
6:4@ (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
∆	variables
«trainable_variables
»regularization_losses
 __call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
д
зtrace_02≈
(__inference_re_lu_10_layer_call_fn_20627Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zзtrace_0
€
иtrace_02а
C__inference_re_lu_10_layer_call_and_return_conditional_losses_20632Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zиtrace_0
0
“0
”1"
trackable_list_wrapper
0
“0
”1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
ћ	variables
Ќtrainable_variables
ќregularization_losses
–__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
е
оtrace_02∆
)__inference_conv2d_11_layer_call_fn_20641Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zоtrace_0
А
пtrace_02б
D__inference_conv2d_11_layer_call_and_return_conditional_losses_20651Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zпtrace_0
*:(@2conv2d_11/kernel
:2conv2d_11/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
’	variables
÷trainable_variables
„regularization_losses
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
я
хtrace_02ј
#__inference_add_layer_call_fn_20657Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zхtrace_0
ъ
цtrace_02џ
>__inference_add_layer_call_and_return_conditional_losses_20663Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zцtrace_0
÷
@0
A1
Z2
[3
t4
u5
О6
П7
®8
©9
¬10
√11
№12
Ё13
ц14
ч15
Р16
С17
™18
Ђ19
ƒ20
≈21"
trackable_list_wrapper
ґ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35"
trackable_list_wrapper
(
ч0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
%__inference_model_layer_call_fn_18947input_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
%__inference_model_layer_call_fn_19088input_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ИBЕ
@__inference_model_layer_call_and_return_conditional_losses_18631input_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ИBЕ
@__inference_model_layer_call_and_return_conditional_losses_18806input_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
џ
е0
ш1
щ2
ъ3
ы4
ь5
э6
ю7
€8
А9
Б10
В11
Г12
Д13
Е14
Ж15
З16
И17
Й18
К19
Л20
М21
Н22
О23
П24
Р25
С26
Т27
У28
Ф29
Х30
Ц31
Ч32
Ш33
Щ34
Ъ35
Ы36
Ь37
Э38
Ю39
Я40
†41
°42
Ґ43
£44
§45
•46
¶47
І48
®49
©50
™51
Ђ52
ђ53
≠54
Ѓ55
ѓ56
∞57
±58
≤59
≥60
і61
µ62
ґ63
Ј64
Є65
є66
Ї67
ї68
Љ69
љ70
Њ71
њ72
ј73
Ѕ74
¬75
√76
ƒ77
≈78
∆79
«80
»81
…82
 83
Ћ84
ћ85
Ќ86
ќ87
ѕ88
–89
—90
“91
”92"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
і
ш0
ъ1
ь2
ю3
А4
В5
Д6
Ж7
И8
К9
М10
О11
Р12
Т13
Ф14
Ц15
Ш16
Ъ17
Ь18
Ю19
†20
Ґ21
§22
¶23
®24
™25
ђ26
Ѓ27
∞28
≤29
і30
ґ31
Є32
Ї33
Љ34
Њ35
ј36
¬37
ƒ38
∆39
»40
 41
ћ42
ќ43
–44
“45"
trackable_list_wrapper
і
щ0
ы1
э2
€3
Б4
Г5
Е6
З7
Й8
Л9
Н10
П11
С12
У13
Х14
Ч15
Щ16
Ы17
Э18
Я19
°20
£21
•22
І23
©24
Ђ25
≠26
ѓ27
±28
≥29
µ30
Ј31
є32
ї33
љ34
њ35
Ѕ36
√37
≈38
«39
…40
Ћ41
Ќ42
ѕ43
—44
”45"
trackable_list_wrapper
э
‘trace_0
’trace_1
÷trace_2
„trace_3
Ўtrace_4
ўtrace_5
Џtrace_6
џtrace_7
№trace_8
Ёtrace_9
ёtrace_10
яtrace_11
аtrace_12
бtrace_13
вtrace_14
гtrace_15
дtrace_16
еtrace_17
жtrace_18
зtrace_19
иtrace_20
йtrace_21
кtrace_22
лtrace_23
мtrace_24
нtrace_25
оtrace_26
пtrace_27
рtrace_28
сtrace_29
тtrace_30
уtrace_31
фtrace_32
хtrace_33
цtrace_34
чtrace_35
шtrace_36
щtrace_37
ъtrace_38
ыtrace_39
ьtrace_40
эtrace_41
юtrace_42
€trace_43
Аtrace_44
Бtrace_452™
"__inference__update_step_xla_19406
"__inference__update_step_xla_19411
"__inference__update_step_xla_19416
"__inference__update_step_xla_19421
"__inference__update_step_xla_19426
"__inference__update_step_xla_19431
"__inference__update_step_xla_19436
"__inference__update_step_xla_19441
"__inference__update_step_xla_19446
"__inference__update_step_xla_19451
"__inference__update_step_xla_19456
"__inference__update_step_xla_19461
"__inference__update_step_xla_19466
"__inference__update_step_xla_19471
"__inference__update_step_xla_19476
"__inference__update_step_xla_19481
"__inference__update_step_xla_19486
"__inference__update_step_xla_19491
"__inference__update_step_xla_19496
"__inference__update_step_xla_19501
"__inference__update_step_xla_19506
"__inference__update_step_xla_19511
"__inference__update_step_xla_19516
"__inference__update_step_xla_19521
"__inference__update_step_xla_19526
"__inference__update_step_xla_19531
"__inference__update_step_xla_19536
"__inference__update_step_xla_19541
"__inference__update_step_xla_19546
"__inference__update_step_xla_19551
"__inference__update_step_xla_19556
"__inference__update_step_xla_19561
"__inference__update_step_xla_19566
"__inference__update_step_xla_19571
"__inference__update_step_xla_19576
"__inference__update_step_xla_19581
"__inference__update_step_xla_19586
"__inference__update_step_xla_19591
"__inference__update_step_xla_19596
"__inference__update_step_xla_19601
"__inference__update_step_xla_19606
"__inference__update_step_xla_19611
"__inference__update_step_xla_19616
"__inference__update_step_xla_19621
"__inference__update_step_xla_19626
"__inference__update_step_xla_19631ѓ
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0z‘trace_0z’trace_1z÷trace_2z„trace_3zЎtrace_4zўtrace_5zЏtrace_6zџtrace_7z№trace_8zЁtrace_9zёtrace_10zяtrace_11zаtrace_12zбtrace_13zвtrace_14zгtrace_15zдtrace_16zеtrace_17zжtrace_18zзtrace_19zиtrace_20zйtrace_21zкtrace_22zлtrace_23zмtrace_24zнtrace_25zоtrace_26zпtrace_27zрtrace_28zсtrace_29zтtrace_30zуtrace_31zфtrace_32zхtrace_33zцtrace_34zчtrace_35zшtrace_36zщtrace_37zъtrace_38zыtrace_39zьtrace_40zэtrace_41zюtrace_42z€trace_43zАtrace_44zБtrace_45
 B«
#__inference_signature_wrapper_19401input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
–BЌ
&__inference_conv2d_layer_call_fn_19640inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
лBи
A__inference_conv2d_layer_call_and_return_conditional_losses_19650inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
3__inference_batch_normalization_layer_call_fn_19663inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
3__inference_batch_normalization_layer_call_fn_19676inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19694inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19712inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
ѕBћ
%__inference_re_lu_layer_call_fn_19717inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
кBз
@__inference_re_lu_layer_call_and_return_conditional_losses_19722inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_1_layer_call_fn_19731inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19741inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_1_layer_call_fn_19754inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_1_layer_call_fn_19767inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19785inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19803inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_1_layer_call_fn_19808inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_1_layer_call_and_return_conditional_losses_19813inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_2_layer_call_fn_19822inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19832inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_2_layer_call_fn_19845inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_2_layer_call_fn_19858inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19876inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19894inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_2_layer_call_fn_19899inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_2_layer_call_and_return_conditional_losses_19904inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_3_layer_call_fn_19913inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19923inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_3_layer_call_fn_19936inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_3_layer_call_fn_19949inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19967inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19985inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_3_layer_call_fn_19990inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_3_layer_call_and_return_conditional_losses_19995inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_4_layer_call_fn_20004inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_4_layer_call_and_return_conditional_losses_20014inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
®0
©1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_4_layer_call_fn_20027inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_4_layer_call_fn_20040inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20058inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20076inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_4_layer_call_fn_20081inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_4_layer_call_and_return_conditional_losses_20086inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_5_layer_call_fn_20095inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_5_layer_call_and_return_conditional_losses_20105inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
¬0
√1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_5_layer_call_fn_20118inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_5_layer_call_fn_20131inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20149inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20167inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_5_layer_call_fn_20172inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_5_layer_call_and_return_conditional_losses_20177inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_6_layer_call_fn_20186inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_6_layer_call_and_return_conditional_losses_20196inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
№0
Ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_6_layer_call_fn_20209inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_6_layer_call_fn_20222inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20240inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20258inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_6_layer_call_fn_20263inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_6_layer_call_and_return_conditional_losses_20268inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_7_layer_call_fn_20277inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_7_layer_call_and_return_conditional_losses_20287inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
ц0
ч1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_7_layer_call_fn_20300inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_7_layer_call_fn_20313inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20331inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20349inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_7_layer_call_fn_20354inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_7_layer_call_and_return_conditional_losses_20359inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_8_layer_call_fn_20368inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_8_layer_call_and_return_conditional_losses_20378inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Р0
С1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_8_layer_call_fn_20391inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_8_layer_call_fn_20404inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20422inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20440inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_8_layer_call_fn_20445inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_8_layer_call_and_return_conditional_losses_20450inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_conv2d_9_layer_call_fn_20459inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_conv2d_9_layer_call_and_return_conditional_losses_20469inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
™0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
5__inference_batch_normalization_9_layer_call_fn_20482inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
5__inference_batch_normalization_9_layer_call_fn_20495inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20513inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЧBФ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20531inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
—Bќ
'__inference_re_lu_9_layer_call_fn_20536inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_re_lu_9_layer_call_and_return_conditional_losses_20541inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
”B–
)__inference_conv2d_10_layer_call_fn_20550inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_10_layer_call_and_return_conditional_losses_20560inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
ƒ0
≈1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
6__inference_batch_normalization_10_layer_call_fn_20573inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
6__inference_batch_normalization_10_layer_call_fn_20586inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20604inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20622inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
“Bѕ
(__inference_re_lu_10_layer_call_fn_20627inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_re_lu_10_layer_call_and_return_conditional_losses_20632inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
”B–
)__inference_conv2d_11_layer_call_fn_20641inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv2d_11_layer_call_and_return_conditional_losses_20651inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
ўB÷
#__inference_add_layer_call_fn_20657inputs_0inputs_1"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
>__inference_add_layer_call_and_return_conditional_losses_20663inputs_0inputs_1"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
В	variables
Г	keras_api

Дtotal

Еcount"
_tf_keras_metric
,:*@2Adam/m/conv2d/kernel
,:*@2Adam/v/conv2d/kernel
:@2Adam/m/conv2d/bias
:@2Adam/v/conv2d/bias
,:*@2 Adam/m/batch_normalization/gamma
,:*@2 Adam/v/batch_normalization/gamma
+:)@2Adam/m/batch_normalization/beta
+:)@2Adam/v/batch_normalization/beta
.:,@@2Adam/m/conv2d_1/kernel
.:,@@2Adam/v/conv2d_1/kernel
 :@2Adam/m/conv2d_1/bias
 :@2Adam/v/conv2d_1/bias
.:,@2"Adam/m/batch_normalization_1/gamma
.:,@2"Adam/v/batch_normalization_1/gamma
-:+@2!Adam/m/batch_normalization_1/beta
-:+@2!Adam/v/batch_normalization_1/beta
.:,@@2Adam/m/conv2d_2/kernel
.:,@@2Adam/v/conv2d_2/kernel
 :@2Adam/m/conv2d_2/bias
 :@2Adam/v/conv2d_2/bias
.:,@2"Adam/m/batch_normalization_2/gamma
.:,@2"Adam/v/batch_normalization_2/gamma
-:+@2!Adam/m/batch_normalization_2/beta
-:+@2!Adam/v/batch_normalization_2/beta
.:,@@2Adam/m/conv2d_3/kernel
.:,@@2Adam/v/conv2d_3/kernel
 :@2Adam/m/conv2d_3/bias
 :@2Adam/v/conv2d_3/bias
.:,@2"Adam/m/batch_normalization_3/gamma
.:,@2"Adam/v/batch_normalization_3/gamma
-:+@2!Adam/m/batch_normalization_3/beta
-:+@2!Adam/v/batch_normalization_3/beta
.:,@@2Adam/m/conv2d_4/kernel
.:,@@2Adam/v/conv2d_4/kernel
 :@2Adam/m/conv2d_4/bias
 :@2Adam/v/conv2d_4/bias
.:,@2"Adam/m/batch_normalization_4/gamma
.:,@2"Adam/v/batch_normalization_4/gamma
-:+@2!Adam/m/batch_normalization_4/beta
-:+@2!Adam/v/batch_normalization_4/beta
.:,@@2Adam/m/conv2d_5/kernel
.:,@@2Adam/v/conv2d_5/kernel
 :@2Adam/m/conv2d_5/bias
 :@2Adam/v/conv2d_5/bias
.:,@2"Adam/m/batch_normalization_5/gamma
.:,@2"Adam/v/batch_normalization_5/gamma
-:+@2!Adam/m/batch_normalization_5/beta
-:+@2!Adam/v/batch_normalization_5/beta
.:,@@2Adam/m/conv2d_6/kernel
.:,@@2Adam/v/conv2d_6/kernel
 :@2Adam/m/conv2d_6/bias
 :@2Adam/v/conv2d_6/bias
.:,@2"Adam/m/batch_normalization_6/gamma
.:,@2"Adam/v/batch_normalization_6/gamma
-:+@2!Adam/m/batch_normalization_6/beta
-:+@2!Adam/v/batch_normalization_6/beta
.:,@@2Adam/m/conv2d_7/kernel
.:,@@2Adam/v/conv2d_7/kernel
 :@2Adam/m/conv2d_7/bias
 :@2Adam/v/conv2d_7/bias
.:,@2"Adam/m/batch_normalization_7/gamma
.:,@2"Adam/v/batch_normalization_7/gamma
-:+@2!Adam/m/batch_normalization_7/beta
-:+@2!Adam/v/batch_normalization_7/beta
.:,@@2Adam/m/conv2d_8/kernel
.:,@@2Adam/v/conv2d_8/kernel
 :@2Adam/m/conv2d_8/bias
 :@2Adam/v/conv2d_8/bias
.:,@2"Adam/m/batch_normalization_8/gamma
.:,@2"Adam/v/batch_normalization_8/gamma
-:+@2!Adam/m/batch_normalization_8/beta
-:+@2!Adam/v/batch_normalization_8/beta
.:,@@2Adam/m/conv2d_9/kernel
.:,@@2Adam/v/conv2d_9/kernel
 :@2Adam/m/conv2d_9/bias
 :@2Adam/v/conv2d_9/bias
.:,@2"Adam/m/batch_normalization_9/gamma
.:,@2"Adam/v/batch_normalization_9/gamma
-:+@2!Adam/m/batch_normalization_9/beta
-:+@2!Adam/v/batch_normalization_9/beta
/:-@@2Adam/m/conv2d_10/kernel
/:-@@2Adam/v/conv2d_10/kernel
!:@2Adam/m/conv2d_10/bias
!:@2Adam/v/conv2d_10/bias
/:-@2#Adam/m/batch_normalization_10/gamma
/:-@2#Adam/v/batch_normalization_10/gamma
.:,@2"Adam/m/batch_normalization_10/beta
.:,@2"Adam/v/batch_normalization_10/beta
/:-@2Adam/m/conv2d_11/kernel
/:-@2Adam/v/conv2d_11/kernel
!:2Adam/m/conv2d_11/bias
!:2Adam/v/conv2d_11/bias
нBк
"__inference__update_step_xla_19406gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19411gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19416gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19421gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19426gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19431gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19436gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19441gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19446gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19451gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19456gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19461gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19466gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19471gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19476gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19481gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19486gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19491gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19496gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19501gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19506gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19511gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19516gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19521gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19526gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19531gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19536gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19541gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19546gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19551gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19556gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19561gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19566gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19571gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19576gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19581gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19586gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19591gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19596gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19601gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19606gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19611gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19616gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19621gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19626gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_19631gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Д0
Е1"
trackable_list_wrapper
.
В	variables"
_generic_user_object
:  (2total
:  (2count§
"__inference__update_step_xla_19406~xҐu
nҐk
!К
gradient@
<Т9	%Ґ"
ъ@
А
p
` VariableSpec 
`јс£ћ¬¬?
™ "
 М
"__inference__update_step_xla_19411f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`арђћ¬¬?
™ "
 М
"__inference__update_step_xla_19416f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јДиА¶¬?
™ "
 М
"__inference__update_step_xla_19421f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јЙиА¶¬?
™ "
 §
"__inference__update_step_xla_19426~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`јШ∞А¶¬?
™ "
 М
"__inference__update_step_xla_19431f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†Ч∞А¶¬?
™ "
 М
"__inference__update_step_xla_19436f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†б≤А¶¬?
™ "
 М
"__inference__update_step_xla_19441f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†ж≤А¶¬?
™ "
 §
"__inference__update_step_xla_19446~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`а≠∞А¶¬?
™ "
 М
"__inference__update_step_xla_19451f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†°∞А¶¬?
™ "
 М
"__inference__update_step_xla_19456f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`аё£ћ¬¬?
™ "
 М
"__inference__update_step_xla_19461f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јз”А¶¬?
™ "
 §
"__inference__update_step_xla_19466~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`†Т∞А¶¬?
™ "
 М
"__inference__update_step_xla_19471f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`АѓиА¶¬?
™ "
 М
"__inference__update_step_xla_19476f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†ТЉА¶¬?
™ "
 М
"__inference__update_step_xla_19481f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†ЧЉА¶¬?
™ "
 §
"__inference__update_step_xla_19486~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`†ГиА¶¬?
™ "
 М
"__inference__update_step_xla_19491f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јЭиА¶¬?
™ "
 М
"__inference__update_step_xla_19496f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†рњА¶¬?
™ "
 М
"__inference__update_step_xla_19501f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†хњА¶¬?
™ "
 §
"__inference__update_step_xla_19506~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`јЙєА¶¬?
™ "
 М
"__inference__update_step_xla_19511f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`аЕєА¶¬?
™ "
 М
"__inference__update_step_xla_19516f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`АС»А¶¬?
™ "
 М
"__inference__update_step_xla_19521f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`АЦ»А¶¬?
™ "
 §
"__inference__update_step_xla_19526~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`†°ЉА¶¬?
™ "
 М
"__inference__update_step_xla_19531f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†ЬЉА¶¬?
™ "
 М
"__inference__update_step_xla_19536f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`анМА¶¬?
™ "
 М
"__inference__update_step_xla_19541f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`атМА¶¬?
™ "
 §
"__inference__update_step_xla_19546~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`А—ƒА¶¬?
™ "
 М
"__inference__update_step_xla_19551f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јќƒА¶¬?
™ "
 М
"__inference__update_step_xla_19556f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јОХА¶¬?
™ "
 М
"__inference__update_step_xla_19561f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јУХА¶¬?
™ "
 §
"__inference__update_step_xla_19566~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`атƒА¶¬?
™ "
 М
"__inference__update_step_xla_19571f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јцƒА¶¬?
™ "
 М
"__inference__update_step_xla_19576f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`АпЩА¶¬?
™ "
 М
"__inference__update_step_xla_19581f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`АфЩА¶¬?
™ "
 §
"__inference__update_step_xla_19586~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`ј…СА¶¬?
™ "
 М
"__inference__update_step_xla_19591f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†“СА¶¬?
™ "
 М
"__inference__update_step_xla_19596f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`аѕ°А¶¬?
™ "
 М
"__inference__update_step_xla_19601f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`а‘°А¶¬?
™ "
 §
"__inference__update_step_xla_19606~xҐu
nҐk
!К
gradient@@
<Т9	%Ґ"
ъ@@
А
p
` VariableSpec 
`аўСА¶¬?
™ "
 М
"__inference__update_step_xla_19611f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`А÷СА¶¬?
™ "
 М
"__inference__update_step_xla_19616f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`арРА¶¬?
™ "
 М
"__inference__update_step_xla_19621f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`Аі¶А¶¬?
™ "
 §
"__inference__update_step_xla_19626~xҐu
nҐk
!К
gradient@
<Т9	%Ґ"
ъ@
А
p
` VariableSpec 
`јз°А¶¬?
™ "
 М
"__inference__update_step_xla_19631f`Ґ]
VҐS
К
gradient
0Т-	Ґ
ъ
А
p
` VariableSpec 
`А«°А¶¬?
™ "
 Ѓ
 __inference__wrapped_model_17593Йv45>?@ANOXYZ[hirstuВГМНОПЬЭ¶І®©ґЈјЅ¬√–—Џџ№ЁклфхцчДЕОПРСЮЯ®©™ЂЄє¬√ƒ≈“”JҐG
@Ґ=
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "C™@
>
add7К4
add+€€€€€€€€€€€€€€€€€€€€€€€€€€€Я
>__inference_add_layer_call_and_return_conditional_losses_20663№СҐН
ЕҐБ
Ъ|
<К9
inputs_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
<К9
inputs_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ щ
#__inference_add_layer_call_fn_20657—СҐН
ЕҐБ
Ъ|
<К9
inputs_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
<К9
inputs_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€ы
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20604•¬√ƒ≈QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ы
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_20622•¬√ƒ≈QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ’
6__inference_batch_normalization_10_layer_call_fn_20573Ъ¬√ƒ≈QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@’
6__inference_batch_normalization_10_layer_call_fn_20586Ъ¬√ƒ≈QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ц
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19785°XYZ[QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ц
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19803°XYZ[QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ –
5__inference_batch_normalization_1_layer_call_fn_19754ЦXYZ[QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@–
5__inference_batch_normalization_1_layer_call_fn_19767ЦXYZ[QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ц
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19876°rstuQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ц
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19894°rstuQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ –
5__inference_batch_normalization_2_layer_call_fn_19845ЦrstuQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@–
5__inference_batch_normalization_2_layer_call_fn_19858ЦrstuQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ъ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19967•МНОПQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ъ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19985•МНОПQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ‘
5__inference_batch_normalization_3_layer_call_fn_19936ЪМНОПQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
5__inference_batch_normalization_3_layer_call_fn_19949ЪМНОПQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ъ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20058•¶І®©QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ъ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20076•¶І®©QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ‘
5__inference_batch_normalization_4_layer_call_fn_20027Ъ¶І®©QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
5__inference_batch_normalization_4_layer_call_fn_20040Ъ¶І®©QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ъ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20149•јЅ¬√QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ъ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_20167•јЅ¬√QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ‘
5__inference_batch_normalization_5_layer_call_fn_20118ЪјЅ¬√QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
5__inference_batch_normalization_5_layer_call_fn_20131ЪјЅ¬√QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ъ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20240•Џџ№ЁQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ъ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20258•Џџ№ЁQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ‘
5__inference_batch_normalization_6_layer_call_fn_20209ЪЏџ№ЁQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
5__inference_batch_normalization_6_layer_call_fn_20222ЪЏџ№ЁQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ъ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20331•фхцчQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ъ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_20349•фхцчQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ‘
5__inference_batch_normalization_7_layer_call_fn_20300ЪфхцчQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
5__inference_batch_normalization_7_layer_call_fn_20313ЪфхцчQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ъ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20422•ОПРСQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ъ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_20440•ОПРСQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ‘
5__inference_batch_normalization_8_layer_call_fn_20391ЪОПРСQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
5__inference_batch_normalization_8_layer_call_fn_20404ЪОПРСQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ъ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20513•®©™ЂQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ъ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_20531•®©™ЂQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ‘
5__inference_batch_normalization_9_layer_call_fn_20482Ъ®©™ЂQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
5__inference_batch_normalization_9_layer_call_fn_20495Ъ®©™ЂQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19694°>?@AQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19712°>?@AQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ќ
3__inference_batch_normalization_layer_call_fn_19663Ц>?@AQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ќ
3__inference_batch_normalization_layer_call_fn_19676Ц>?@AQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@в
D__inference_conv2d_10_layer_call_and_return_conditional_losses_20560ЩЄєIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Љ
)__inference_conv2d_10_layer_call_fn_20550ОЄєIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@в
D__inference_conv2d_11_layer_call_and_return_conditional_losses_20651Щ“”IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Љ
)__inference_conv2d_11_layer_call_fn_20641О“”IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€я
C__inference_conv2d_1_layer_call_and_return_conditional_losses_19741ЧNOIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ є
(__inference_conv2d_1_layer_call_fn_19731МNOIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@я
C__inference_conv2d_2_layer_call_and_return_conditional_losses_19832ЧhiIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ є
(__inference_conv2d_2_layer_call_fn_19822МhiIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
C__inference_conv2d_3_layer_call_and_return_conditional_losses_19923ЩВГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
(__inference_conv2d_3_layer_call_fn_19913ОВГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
C__inference_conv2d_4_layer_call_and_return_conditional_losses_20014ЩЬЭIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
(__inference_conv2d_4_layer_call_fn_20004ОЬЭIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
C__inference_conv2d_5_layer_call_and_return_conditional_losses_20105ЩґЈIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
(__inference_conv2d_5_layer_call_fn_20095ОґЈIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
C__inference_conv2d_6_layer_call_and_return_conditional_losses_20196Щ–—IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
(__inference_conv2d_6_layer_call_fn_20186О–—IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
C__inference_conv2d_7_layer_call_and_return_conditional_losses_20287ЩклIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
(__inference_conv2d_7_layer_call_fn_20277ОклIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
C__inference_conv2d_8_layer_call_and_return_conditional_losses_20378ЩДЕIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
(__inference_conv2d_8_layer_call_fn_20368ОДЕIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
C__inference_conv2d_9_layer_call_and_return_conditional_losses_20469ЩЮЯIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
(__inference_conv2d_9_layer_call_fn_20459ОЮЯIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ё
A__inference_conv2d_layer_call_and_return_conditional_losses_19650Ч45IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ј
&__inference_conv2d_layer_call_fn_19640М45IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ў
@__inference_model_layer_call_and_return_conditional_losses_18631Фv45>?@ANOXYZ[hirstuВГМНОПЬЭ¶І®©ґЈјЅ¬√–—Џџ№ЁклфхцчДЕОПРСЮЯ®©™ЂЄє¬√ƒ≈“”RҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
@__inference_model_layer_call_and_return_conditional_losses_18806Фv45>?@ANOXYZ[hirstuВГМНОПЬЭ¶І®©ґЈјЅ¬√–—Џџ№ЁклфхцчДЕОПРСЮЯ®©™ЂЄє¬√ƒ≈“”RҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≥
%__inference_model_layer_call_fn_18947Йv45>?@ANOXYZ[hirstuВГМНОПЬЭ¶І®©ґЈјЅ¬√–—Џџ№ЁклфхцчДЕОПРСЮЯ®©™ЂЄє¬√ƒ≈“”RҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
%__inference_model_layer_call_fn_19088Йv45>?@ANOXYZ[hirstuВГМНОПЬЭ¶І®©ґЈјЅ¬√–—Џџ№ЁклфхцчДЕОПРСЮЯ®©™ЂЄє¬√ƒ≈“”RҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€џ
C__inference_re_lu_10_layer_call_and_return_conditional_losses_20632УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ µ
(__inference_re_lu_10_layer_call_fn_20627ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_1_layer_call_and_return_conditional_losses_19813УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_1_layer_call_fn_19808ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_2_layer_call_and_return_conditional_losses_19904УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_2_layer_call_fn_19899ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_3_layer_call_and_return_conditional_losses_19995УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_3_layer_call_fn_19990ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_4_layer_call_and_return_conditional_losses_20086УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_4_layer_call_fn_20081ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_5_layer_call_and_return_conditional_losses_20177УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_5_layer_call_fn_20172ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_6_layer_call_and_return_conditional_losses_20268УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_6_layer_call_fn_20263ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_7_layer_call_and_return_conditional_losses_20359УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_7_layer_call_fn_20354ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_8_layer_call_and_return_conditional_losses_20450УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_8_layer_call_fn_20445ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
B__inference_re_lu_9_layer_call_and_return_conditional_losses_20541УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
'__inference_re_lu_9_layer_call_fn_20536ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ў
@__inference_re_lu_layer_call_and_return_conditional_losses_19722УIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "FҐC
<К9
tensor_0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≤
%__inference_re_lu_layer_call_fn_19717ИIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ ";К8
unknown+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Љ
#__inference_signature_wrapper_19401Фv45>?@ANOXYZ[hirstuВГМНОПЬЭ¶І®©ґЈјЅ¬√–—Џџ№ЁклфхцчДЕОПРСЮЯ®©™ЂЄє¬√ƒ≈“”UҐR
Ґ 
K™H
F
input_1;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€"C™@
>
add7К4
add+€€€€€€€€€€€€€€€€€€€€€€€€€€€