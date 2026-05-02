unit uLlamaCpp;

interface

uses
  Classes, uBaseLibrary, uLLMBaseTypes;

type
  TLlamaAttentionType = (
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1
  );

  TLlamaFlashAttentionType = (
    LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
    LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1
  );

  TLlamaSplitMode = (
    LLAMA_SPLIT_MODE_NONE  = 0, // single GPU
    LLAMA_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
    LLAMA_SPLIT_MODE_ROW   = 2  // split rows across GPUs
  );

  TLlamaRopeScalingType = (
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = Integer(LLAMA_ROPE_SCALING_TYPE_LONGROPE)
  );

  TLlamaPoolingType = (
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE = 0,
    LLAMA_POOLING_TYPE_MEAN = 1,
    LLAMA_POOLING_TYPE_CLS  = 2,
    LLAMA_POOLING_TYPE_LAST = 3,
    LLAMA_POOLING_TYPE_RANK = 4  // used by reranking models to attach the classification head to the graph
  );

  TGgmlLogLevel = (
    GGML_LOG_LEVEL_NONE  = 0,
    GGML_LOG_LEVEL_DEBUG = 1,
    GGML_LOG_LEVEL_INFO  = 2,
    GGML_LOG_LEVEL_WARN  = 3,
    GGML_LOG_LEVEL_ERROR = 4,
    GGML_LOG_LEVEL_CONT  = 5
  );

  TGGMLNumaStrategy = (
    GGML_NUMA_STRATEGY_DISABLED   = 0,
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
    GGML_NUMA_STRATEGY_ISOLATE    = 2,
    GGML_NUMA_STRATEGY_NUMACTL    = 3,
    GGML_NUMA_STRATEGY_MIRROR     = 4,
    GGML_NUMA_STRATEGY_COUNT
  );

  TGGMLType = (
    GGML_TYPE_F32 = 0, 
    GGML_TYPE_F16 = 1, 
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q4_2 = 4, // support has been removed
    GGML_TYPE_Q4_3 = 5, // support has been removed
    GGML_TYPE_Q5_0 = 6, 
    GGML_TYPE_Q5_1 = 7, 
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9, 
    GGML_TYPE_Q2_K = 10, 
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12, 
    GGML_TYPE_Q5_K = 13, 
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15, 
    GGML_TYPE_IQ2_XXS = 16, 
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18, 
    GGML_TYPE_IQ1_S = 19, 
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21, 
    GGML_TYPE_IQ2_S = 22, 
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24, 
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27, 
    GGML_TYPE_F64 = 28, 
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30, 
    GGML_TYPE_Q4_0_4_4 = 31, 
    GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33, 
    GGML_TYPE_TQ1_0 = 34, 
    GGML_TYPE_TQ2_0 = 35,
    GGML_TYPE_COUNT = 36 // Number of types (excluding the commented ones)
  );

  TLlamaModelParams = class(TObject)
  public
    constructor Create(AJSON: JSONvariant); overload;
  published
    devices: Integer;
    tensor_buft_overrides: Integer;
    n_gpu_layers: Integer;
    split_mode: TLlamaSplitMode;
    main_gpu: Integer;
    tensor_split: Integer;
    llama_model_kv_override: Integer;
    vocab_only: Boolean;
    use_mmap: Boolean;
    use_direct_io: Boolean;
    use_mlock: Boolean;
    check_tensors: Boolean;
    use_extra_bufts: Boolean;
    no_host: Boolean;
    no_alloc: Boolean;
  end;

  TLlamaContextParams = class(TObject)
  public
    constructor Create(AJSON: JSONvariant); overload;
  published
    n_ctx: Integer;
    n_batch: Integer;
    n_ubatch: Integer;
    n_seq_max: Integer;
    n_threads: Integer;
    n_threads_batch: Integer;
    rope_scaling_type: TLlamaRopeScalingType;
    pooling_type: TLlamaPoolingType;
    llama_attention_type: TLlamaAttentionType;
    llama_flash_attn_type: TLlamaFlashAttentionType;
    rope_freq_base: Float;
    rope_freq_scale: Float;
    yarn_ext_factor: Float;
    yarn_attn_factor: Float;
    yarn_beta_fast: Float;
    yarn_beta_slow: Float;
    yarn_orig_ctx: Integer;
    defrag_thold: Float;
    type_k: Integer;
    type_v: Integer;
    embeddings: Boolean;
    offload_kqv: Boolean;
    no_perf: Boolean;
    op_offload: Boolean;
    swa_full: Boolean;
    kv_unified: Boolean;
    samplers: Integer;
    n_samplers: Integer;
  end;

  TCompletationParams = class(TObject)
  public
    constructor Create(AJSON: JSONvariant); overload;
  published
    MinGenTokens: Integer;
    MaxGenTokens: Integer;
    Temperature: Float;
    TopK: Integer;
    TopP: Float;
    RepeatPenalty: Float;
    RepeatLastN: Integer;
  end;

  TLlamaSettings = class(TObject)
  public
    constructor Create(AJSON: JSONvariant); overload;
    function ToString: string;
  published
    LlamaModelParams: TLlamaModelParams;
    LlamaContextParams: TLlamaContextParams;
    CompletationParams: TCompletationParams;
  end;

  TLLamaLogCallback = procedure(Sender: TObject; const LogLevel: TGgmlLogLevel; const ALog: string) of object;
  TLLamaLogCallback2 = procedure(const LogLevel: TGgmlLogLevel; const ALog: string);
  TLlamaTokenCallback = procedure(Sender: TObject; const AToken: integer; const ATextToken: string; var AContinue: boolean) of object;
  TLlamaTokenCallback2 = procedure(const AToken: integer; const ATextToken: string; var AContinue: boolean);
  TLlamaNotifyEvent = procedure(Sender: TObject) of object;
  TLlamaNotifyEvent2 = procedure();

  TLlama=class(TObject)
  private
    FKeyInstance: integer;
    FSettings: TLlamaSettings;
    FOnLogCallback: TLLamaLogCallback;
    FOnLogCallback2: TLLamaLogCallback2;
    FOnTokenCallback: TLlamaTokenCallback;
    FOnTokenCallback2: TLlamaTokenCallback2;
    FOnGenerationComplete: TLlamaNotifyEvent;
    FOnGenerationComplete2: TLlamaNotifyEvent2;
    function GetModelPath: string;
  public
    constructor Create;
    destructor Destroy; override;
    procedure LoadModel(const AModel: string);
    procedure UnloadModel;
    function Tokenize(const Text: string; AddSpecial:Boolean=true; ParseSpecial: Boolean=true): TIntegerArray;
    function Detokenize(Token: integer; IncludeSpecial: Boolean = true): string; overload;
    function Detokenize(Tokens: TIntegerArray; IncludeSpecial: Boolean = true): string; overload;
    function StartCompletation(const Prompt: string): string;
    procedure StartChatCompletation(var InfoChat: TInfoChat);
    property ModelPath: string read GetModelPath;
    property Settings: TLlamaSettings read FSettings write FSettings;
    property OnLogCallback: TLLamaLogCallback read FOnLogCallback write FOnLogCallback;
    property OnLogCallback2: TLLamaLogCallback2 read FOnLogCallback2 write FOnLogCallback2;
    property OnTokenCallback: TLlamaTokenCallback read FOnTokenCallback write FOnTokenCallback;
    property OnTokenCallback2: TLlamaTokenCallback2 read FOnTokenCallback2 write FOnTokenCallback2;
    property OnGenerationComplete: TLlamaNotifyEvent read FOnGenerationComplete write FOnGenerationComplete; 
    property OnGenerationComplete2: TLlamaNotifyEvent2 read FOnGenerationComplete2 write FOnGenerationComplete2; 
  end;

var 
  LLama: TLlama;

implementation

const
  LibGUID='{50EE9228-2C34-4A97-B014-0AEF14FF8DE6}';

var
  LlamaInstances: array[integer] of TLlama;
  KeyCounter: integer=0;
  InternalModelPath: string:='';

{ Other procedures }

procedure LogCallback(const Parameters: TVariantArray);
begin
  for var Key in LlamaInstances.Keys do
    begin
      if Assigned(LlamaInstances[Key].OnLogCallback) then
        LlamaInstances[Key].OnLogCallback(LlamaInstances[Key], TGgmlLogLevel(Parameters[1]), Parameters[0]);
      if Assigned(LlamaInstances[Key].OnLogCallback2) then
        LlamaInstances[Key].OnLogCallback2(TGgmlLogLevel(Parameters[1]), Parameters[0]);
    end;
end;

procedure TokenCallback(const Parameters: TVariantArray);
var 
  FlagContinue: boolean;
  Args: TVariantArray;
begin
  FlagContinue:=true;
  for var Key in LlamaInstances.Keys do
    begin
      if Assigned(LlamaInstances[Key].OnTokenCallback) then
        LlamaInstances[Key].OnTokenCallback(LlamaInstances[Key],Parameters[0],Parameters[1],FlagContinue);
      if (FlagContinue) then
        if Assigned(LlamaInstances[Key].OnTokenCallback2) then
          LlamaInstances[Key].OnTokenCallback2(Parameters[0],Parameters[1],FlagContinue);
      if (not FlagContinue) then
        Break;
    end;
  if (not FlagContinue) then
    __LibInterface_InvokeLibProc(LibGUID, 0, 'TCC', Args);
end;

procedure GenerationCompleteCallback(const Parameters: TVariantArray);
begin
  for var Key in LlamaInstances.Keys do
    begin
      if Assigned(LlamaInstances[Key].OnGenerationComplete) then
        LlamaInstances[Key].OnGenerationComplete(LlamaInstances[Key]);
      if Assigned(LlamaInstances[Key].OnGenerationComplete2) then
        LlamaInstances[Key].OnGenerationComplete2();
    end;
end;

{ TLlamaModelParams }

constructor TLlamaModelParams.Create(AJSON: JSONvariant);
begin
  Create;
  devices:=VarToInt(AJSON.devices);
  tensor_buft_overrides:=VarToInt(AJSON.tensor_buft_overrides);
  n_gpu_layers:=VarToInt(AJSON.n_gpu_layers);
  split_mode:=TLlamaSplitMode(VarToInt(AJSON.split_mode));
  main_gpu:=VarToInt(AJSON.main_gpu);
  tensor_split:=VarToInt(AJSON.tensor_split);
  llama_model_kv_override:=VarToInt(AJSON.llama_model_kv_override);
  vocab_only:=VarToBool(AJSON.vocab_only);
  use_mmap:=VarToBool(AJSON.use_mmap);
  use_direct_io:=VarToBool(AJSON.use_direct_io);
  use_mlock:=VarToBool(AJSON.use_mlock);
  check_tensors:=VarToBool(AJSON.check_tensors);
  use_extra_bufts:=VarToBool(AJSON.use_extra_bufts);
  no_host:=VarToBool(AJSON.no_host);
  no_alloc:=VarToBool(AJSON.no_alloc);
end;

{ TLlamaContextParams }

constructor TLlamaContextParams.Create(AJSON: JSONvariant);
begin
  Create;
  n_ctx:=VarToInt(AJSON.n_ctx);
  n_batch:=VarToInt(AJSON.n_batch);
  n_ubatch:=VarToInt(AJSON.n_ubatch);
  n_seq_max:=VarToInt(AJSON.n_seq_max);
  n_threads:=VarToInt(AJSON.n_threads);
  n_threads_batch:=VarToInt(AJSON.n_threads_batch);
  rope_scaling_type:=TLlamaRopeScalingType(VarToInt(AJSON.rope_scaling_type));
  pooling_type:=TLlamaPoolingType(VarToInt(AJSON.pooling_type));
  llama_attention_type:=TLlamaAttentionType(VarToInt(AJSON.llama_attention_type));
  llama_flash_attn_type:=TLlamaFlashAttentionType(VarToInt(AJSON.llama_flash_attn_type));
  rope_freq_base:=VarToFloat(AJSON.rope_freq_base);
  rope_freq_scale:=VarToFloat(AJSON.rope_freq_scale);
  yarn_ext_factor:=VarToFloat(AJSON.yarn_ext_factor);
  yarn_attn_factor:=VarToFloat(AJSON.yarn_attn_factor);
  yarn_beta_fast:=VarToFloat(AJSON.yarn_beta_fast);
  yarn_beta_slow:=VarToFloat(AJSON.yarn_beta_slow);
  yarn_orig_ctx:=VarToInt(AJSON.yarn_orig_ctx);
  defrag_thold:=VarToFloat(AJSON.defrag_thold);
  type_k:=VarToInt(AJSON.type_k);
  type_v:=VarToInt(AJSON.type_v);
  embeddings:=VarToBool(AJSON.embeddings);
  offload_kqv:=VarToBool(AJSON.offload_kqv);
  no_perf:=VarToBool(AJSON.no_perf);
  op_offload:=VarToBool(AJSON.op_offload);
  swa_full:=VarToBool(AJSON.swa_full);
  kv_unified:=VarToBool(AJSON.kv_unified);
  samplers:=VarToInt(AJSON.samplers);
  n_samplers:=VarToInt(AJSON.n_samplers);
end;

{ TCompletationParams }

constructor TCompletationParams.Create(AJSON: JSONvariant);
begin
  Create;
  MinGenTokens:=VarToInt(AJSON.MinGenTokens);
  MaxGenTokens:=VarToInt(AJSON.MaxGenTokens);
  Temperature:=VarToFloat(AJSON.Temperature);
  TopK:=VarToInt(AJSON.TopK);
  TopP:=VarToFloat(AJSON.TopP);
  RepeatPenalty:=VarToFloat(AJSON.RepeatPenalty);
end;

{ TLlamaSettings }

constructor TLlamaSettings.Create(AJSON: JSONvariant);
begin
  Create;
  LlamaModelParams:=TLlamaModelParams.Create(AJSON.LlamaModelParams);
  LlamaContextParams:=TLlamaContextParams.Create(AJSON.LlamaContextParams);
  CompletationParams:=TCompletationParams.Create(AJSON.CompletationParams);
end;

function TLlamaSettings.ToString: string;
begin
  Result:=JSON.Stringify(Self);
end;

{ TLlama }

constructor TLlama.Create;
var
  Args: TVariantArray;
begin
  inherited Create;
  FSettings:=TLlamaSettings.Create(
    JSON.Parse(__LibInterface_InvokeLibProc(LibGUID, 0, 'RLS', Args))
  );
  FKeyInstance:=KeyCounter;
  Inc(KeyCounter);
  LlamaInstances[FKeyInstance]:=Self;
end;

destructor TLlama.Destroy;
begin
  LlamaInstances.Delete(FKeyInstance);
  inherited Destroy;
end;

function TLlama.GetModelPath: string;
begin
  Result:=InternalModelPath;
end;

procedure TLlama.LoadModel(const AModel: string);
var
  Args: TVariantArray;
begin
  Args.Push(AModel);
  Args.Push(FSettings.ToString);
  try
    __LibInterface_InvokeLibProc(LibGUID, 0, 'LMO', Args);
    InternalModelPath:=AModel;
  finally
    ProcessPostedMessage;
  end;
  Args:=[''];
  repeat
    ProcessPostedMessage;
    Sleep(250);
  until (__LibInterface_InvokeLibProc(LibGUID, 0, 'TTE', Args));
  ProcessPostedMessage;
end;

procedure TLlama.UnloadModel;
var
  Args: TVariantArray;
begin
  __LibInterface_InvokeLibProc(LibGUID, 0, 'ULMO', Args);
  InternalModelPath:='';
end;

function TLlama.StartCompletation(const Prompt: string): string;
var
  Args: TVariantArray;
begin
  Args.Push(Prompt);
  Args.Push(FSettings.ToString);
  try
    __LibInterface_InvokeLibProc(LibGUID, 0, 'SC', Args);
  finally
    ProcessPostedMessage;
  end;
  Args:=[''];
  repeat
    ProcessPostedMessage;
    Sleep(250);
  until (__LibInterface_InvokeLibProc(LibGUID, 0, 'TTE', Args));
  Result:=Args[0];
  ProcessPostedMessage;
end;

procedure TLlama.StartChatCompletation(var InfoChat: TInfoChat);
var
  Args: TVariantArray;
begin
  Args.Push(InfoChat.ToString);
  Args.Push(FSettings.ToString);
  __LibInterface_InvokeLibProc(LibGUID, 0, 'SCC', Args);
  Args:=[''];
  repeat
    ProcessPostedMessage;
    Sleep(250);
  until (__LibInterface_InvokeLibProc(LibGUID, 0, 'TTE', Args));
  ProcessPostedMessage;
  InfoChat.AddMessage(rmcAssistant, Args[0]);
end;

function TLlama.Tokenize(const Text: string; AddSpecial,ParseSpecial: Boolean): array of integer;
var
  Args: TVariantArray;
begin
  Args.Push(Text);
  Args.Push(AddSpecial);
  Args.Push(ParseSpecial);
  Result:=__VariantToArray(__LibInterface_InvokeLibProc(LibGUID, 0, 'TKN', Args)).Map(VarToInt);
end;

function TLlama.Detokenize(Token: integer; IncludeSpecial: Boolean): string;
var
  Args: TVariantArray;
begin
  Args.Push(Token);
  Args.Push(IncludeSpecial);
  Result:=__LibInterface_InvokeLibProc(LibGUID, 0, 'DTK', Args);
end;

function TLlama.Detokenize(Tokens: TIntegerArray; IncludeSpecial: Boolean): string;
var
  Args: TVariantArray;
begin
  Args.Push(__ArrayToVariant(Tokens));
  Args.Push(IncludeSpecial);
  Result:=__LibInterface_InvokeLibProc(LibGUID, 0, 'DTKS', Args);
end;

initialization
  Randomize;
  RegisterMessageCallback('log_'+LibGUID,LogCallback);
  RegisterMessageCallback('token_'+LibGUID,TokenCallback);
  RegisterMessageCallback('gc_'+LibGUID,GenerationCompleteCallback);
finalization
  LlamaInstances.Clear;
end.
