library LLamaCppLib;

uses
  System.SysUtils,
  System.JSON,
  System.Classes,
  System.StrUtils,
  System.Variants,
  System.Rtti,
  System.Types,
  System.TypInfo,
  System.IOUtils,
  System.SyncObjs,
  Generics.Collections,
  uLibInterface,
  uUtility,
  uLlamaCPPBinding,
  uLlamaCPPLibBase,
  uLlamaCPPCompletation,
  uLlamaCPPEmbeddings;

{$R *.res}

const
  InternalLibGUID='{50EE9228-2C34-4A97-B014-0AEF14FF8DE6}';

var
  LibFreeing: boolean;
  CallerContext: NativeInt;
  InvokeHost: TInvokeHostProc;
  Task: TThread;
  TaskResult: string;
  ModelInfo: TModelInfo;

type
  TLlamaSettings = record
  public
    LlamaModelParams: TLlamaInstance.llama_model_params;
    LlamaContextParams: TLlamaInstance.llama_context_params;
    CompletationParams: TCompletationParams;
    constructor Create(AJSONString: string); overload;
    constructor Create(AJSON: TJSONObject); overload;
    function ToJSON: TJSONObject;
  end;

  TTaskBase=class(TThread)
  protected
  end;

  TTaskLoadModel=class(TTaskBase)
  private
    FModelName: string;
    FLlamaSettings: TLlamaSettings;
  protected
    procedure Execute; override;
  public
    constructor Create(AModelName: string; ALlamaSettings: TLlamaSettings);
  end;

  TTaskCompletation=class(TTaskBase)
  private
    FModelInfo: TModelInfo;
    FPrompt: string;
    FState: TChatState;
    FCompletationParams: TCompletationParams;
  protected
    procedure Execute; override;
  public
    constructor Create(
      AModelInfo: TModelInfo;
      ACompletationParams: TCompletationParams;
      APrompt: string
    );
  end;

  TTaskChatCompletation=class(TTaskBase)
  private
    FModelInfo: TModelInfo;
    FChatMessageArray: TArray<TChatMessage>;
    FState: TChatState;
    FCompletationParams: TCompletationParams;
  protected
    procedure Execute; override;
  public
    constructor Create(
      AModelInfo: TModelInfo;
      ACompletationParams: TCompletationParams;
      AChatMessageArray: TArray<TChatMessage>
    );
  end;

procedure LogCallback(Level: TGgmlLogLevel; const Text: PAnsiChar; user_data: Pointer); cdecl;
var
  Args: array of variant;
begin
  if LibFreeing then
    Exit;
  Args:=[
    WideString('log_'+InternalLibGUID),
    WideString(string(Text)),
    Ord(Level)
  ];
  InvokeHost(CallerContext, COMHOST_POSTMESSAGE, Args);
end;

procedure TokenCallback(
  const Token: Integer;
  const TokenText: string;
  var FlagContinue: Boolean);
var
  Args: array of variant;
begin
  if LibFreeing then
    Exit;
  Args:=[
    WideString('token_'+InternalLibGUID),
    Token,
    WideString(TokenText)
  ];
  InvokeHost(CallerContext, COMHOST_POSTMESSAGE, Args);
  FlagContinue:=(Assigned(Task) and (not Task.Finished));
end;

function LlamaSettingsToJSONString: string;
var
  LlamaSettings: TLlamaSettings;
  Data: TJSONObject;
begin
  with LlamaSettings do
    begin
      LlamaModelParams:=LoadModelParams;
      LlamaContextParams:=LoadContextParams;
      with CompletationParams do
        begin
          MinGenTokens:=128;
          MaxGenTokens:=2000;
          Temperature := 0.7;
          TopK := 40;
          TopP := 0.95;
          RepeatPenalty := 1.1;
          RepeatLastN := 64;
        end;
    end;
  Data:=LlamaSettings.ToJSON;
  try
    Result:=Data.ToString;
  finally
    Data.Free;
  end;
end;

function DecodeInfoChat(const AString: string): TArray<TChatMessage>;
var
  InfoChat: TJSONObject;
  MessageArray: TJSONArray;
begin
  Result:=[];
  InfoChat:=ParseJSONObject(AString);
  try
    MessageArray:=ReadJSONArray(InfoChat,'Messages');
    for var AMessage in MessageArray do
      begin
        var ChatMessage:=Default(TChatMessage);
        ChatMessage.Role:=TChatRole(VarToInt(ReadJSONValue(AMessage,'Role',0)));
        ChatMessage.Content:=ReadJSONValue(AMessage,'Content','');
        Result:=Result+[ChatMessage];
      end;
  finally
    InfoChat.Free;
  end;
end;

procedure InternalUnloadModel;
begin
  with ModelInfo do
    if Assigned(Model) or Assigned(Context) then
      UnloadModel(ModelInfo);
end;

function InvokeLibProcImpl(Context, Instance: NativeInt; const ProcName: PChar; var Args:array of variant): Variant; cdecl;
begin
  if (SameText(ProcName,'LMO')) then
    begin
      InternalUnloadModel;
      Task:=TTaskLoadModel.Create(
        Args[0],
        TLlamaSettings.Create(Args[1])
      );
    end
  else if (SameText(ProcName,'ULMO')) then
    InternalUnloadModel
  else if (SameText(ProcName,'SC')) then
    begin
      var LlamaSettings: TLlamaSettings:=TLlamaSettings.Create(Args[1]);
      Task:=TTaskCompletation.Create(
        ModelInfo,
        LlamaSettings.CompletationParams,
        Args[0]
      );
    end
  else if (SameText(ProcName,'SCC')) then
    begin
      var ChatMessageArray: TArray<TChatMessage>:=DecodeInfoChat(Args[0]);
      var LlamaSettings: TLlamaSettings:=TLlamaSettings.Create(Args[1]);
      Task:=TTaskChatCompletation.Create(
        ModelInfo,
        LlamaSettings.CompletationParams,
        ChatMessageArray
      );
    end
  else if (SameText(ProcName,'TCC')) then
    begin
      if (Assigned(Task)) then
        begin
          Task.Terminate;
          Task:=nil;
        end;
    end
  else if (SameText(ProcName,'TTE')) then
    begin
      Result:=(not Assigned(Task)) or Task.Finished;
      if (Result) then
        begin
          Task:=nil;
          Args[0]:=TaskResult;
        end;
    end
  else if (SameText(ProcName,'RLS')) then
    Result:=WideString(LlamaSettingsToJSONString)
  else if (SameText(ProcName,'TKN')) then
    begin
      var Tokens: TArray<Integer>:=Tokenize(
        ModelInfo,
        Args[0],
        VarToBoolean(Args[1]),
        VarToBoolean(Args[2])
      );
      DynArrayToVariant(Result,Pointer(Tokens),TypeInfo(TArray<Integer>));
    end
  else if (SameText(ProcName,'DTK')) then
    Result:=WideString(TokenToString(
      ModelInfo,
      Args[0],
      VarToBoolean(Args[1])
    ))
  else if (SameText(ProcName,'DTKS')) then
    begin
      var Dati: TArray<integer>;
      DynArrayFromVariant(Pointer(Dati),Args[0],TypeInfo(TArray<integer>));
      Result:=WideString(TokensToString(
        ModelInfo,
        Dati,
        VarToBoolean(Args[1])
      ));
    end
  else if (SameText(ProcName,'TKNEMB')) then
    begin
      var Vector: TArray<Single> := GetTokenEmbedding(
        ModelInfo,
        Args[0]
      );
      DynArrayToVariant(Result,Pointer(Vector),TypeInfo(TArray<Single>));
    end
  else if (SameText(ProcName,'VOCSIZE')) then
    Result:=IntToStr(GetVocabSize(ModelInfo))
  else
    RaiseException('Proc "%s" unknown',[ProcName]);
end;

procedure LibInit(const ALibInterface: PLibInterface); cdecl;
begin
  LibFreeing:=false;
  with ALibInterface^ do
    begin
      LibGUID:=InternalLibGUID;
      InvokeLibProc:=InvokeLibProcImpl;
      CallerContext:=Context;
      InvokeHost:=InvokeHostProc;
      with LLamaInstance do
        begin
          Load(ExecutionPath+'llamacpp');
          AddLogCallback(LogCallback,nil);
          Init;
        end;
    end;
  ModelInfo:=Default(TModelInfo);
end;

procedure LibFree(const ALibInterface: PLibInterface); cdecl;
begin
  LibFreeing:=true;
  Task:=nil;
  InternalUnloadModel;
  LLamaInstance.UnLoad;
end;

{ TLlamaSettings }

constructor TLlamaSettings.Create(AJSONString: string);
var Data: TJSONObject;
begin
  Data:=ParseJSONObject(AJSONString);
  try
    Create(Data);
  finally
    Data.Free;
  end;
end;

constructor TLlamaSettings.Create(AJSON: TJSONObject);
var
  SubObject: TJSONObject;
begin
  LlamaModelParams:=Default(TLlamaInstance.llama_model_params);
  LlamaContextParams:=Default(TLlamaInstance.llama_context_params);

  SubObject:=ReadJSONObject(AJSON, 'LlamaModelParams');
  LlamaModelParams.devices:=pointer(ReadJSONValue(SubObject,'devices',integer(LlamaModelParams.devices)));
  LlamaModelParams.tensor_buft_overrides:=pointer(ReadJSONValue(SubObject,'tensor_buft_overrides',integer(LlamaModelParams.tensor_buft_overrides)));
  LlamaModelParams.n_gpu_layers:=ReadJSONValue(SubObject,'n_gpu_layers',LlamaModelParams.n_gpu_layers);
  LlamaModelParams.split_mode:=TLlamaSplitMode(ReadJSONValue(SubObject,'split_mode',Ord(LlamaModelParams.split_mode)));
  LlamaModelParams.main_gpu:=ReadJSONValue(SubObject,'main_gpu',LlamaModelParams.main_gpu);
  LlamaModelParams.tensor_split:=pointer(ReadJSONValue(SubObject,'tensor_split',integer(LlamaModelParams.tensor_split)));
  LlamaModelParams.llama_model_kv_override:=pointer(ReadJSONValue(SubObject,'llama_model_kv_override',integer(LlamaModelParams.llama_model_kv_override)));
  LlamaModelParams.vocab_only:=ReadJSONValue(SubObject,'vocab_only',LlamaModelParams.vocab_only);
  LlamaModelParams.use_mmap:=ReadJSONValue(SubObject,'use_mmap',LlamaModelParams.use_mmap);
  LlamaModelParams.use_direct_io:=ReadJSONValue(SubObject,'use_direct_io',LlamaModelParams.use_direct_io);
  LlamaModelParams.use_mlock:=ReadJSONValue(SubObject,'use_mlock',LlamaModelParams.use_mlock);
  LlamaModelParams.check_tensors:=ReadJSONValue(SubObject,'check_tensors',LlamaModelParams.check_tensors);
  LlamaModelParams.use_extra_bufts:=ReadJSONValue(SubObject,'use_extra_bufts',LlamaModelParams.use_extra_bufts);
  LlamaModelParams.no_host:=ReadJSONValue(SubObject,'no_host',LlamaModelParams.no_host);
  LlamaModelParams.no_alloc:=ReadJSONValue(SubObject,'no_alloc',LlamaModelParams.no_alloc);

  SubObject:=ReadJSONObject(AJSON, 'LlamaContextParams');
  LlamaContextParams.n_ctx:=ReadJSONValue(SubObject,'n_ctx',LlamaContextParams.n_ctx);
  LlamaContextParams.n_batch:=ReadJSONValue(SubObject,'n_batch',LlamaContextParams.n_batch);
  LlamaContextParams.n_ubatch:=ReadJSONValue(SubObject,'n_ubatch',LlamaContextParams.n_ubatch);
  LlamaContextParams.n_seq_max:=ReadJSONValue(SubObject,'n_seq_max',LlamaContextParams.n_seq_max);
  LlamaContextParams.n_threads:=ReadJSONValue(SubObject,'n_threads',LlamaContextParams.n_threads);
  LlamaContextParams.n_threads_batch:=ReadJSONValue(SubObject,'n_threads_batch',LlamaContextParams.n_threads_batch);
  LlamaContextParams.rope_scaling_type:=TLlamaRopeScalingType(ReadJSONValue(SubObject,'rope_scaling_type',Ord(LlamaContextParams.rope_scaling_type)));
  LlamaContextParams.pooling_type:=TLlamaPoolingType(ReadJSONValue(SubObject,'pooling_type',integer(LlamaContextParams.pooling_type)));
  LlamaContextParams.llama_attention_type:=TLlamaAttentionType(ReadJSONValue(SubObject,'llama_attention_type',integer(LlamaContextParams.llama_attention_type)));
  LlamaContextParams.llama_flash_attn_type:=TLlamaFlashAttentionType(ReadJSONValue(SubObject,'llama_flash_attn_type',integer(LlamaContextParams.llama_flash_attn_type)));
  LlamaContextParams.rope_freq_base:=ReadJSONValue(SubObject,'rope_freq_base',LlamaContextParams.rope_freq_base);
  LlamaContextParams.rope_freq_scale:=ReadJSONValue(SubObject,'rope_freq_scale',LlamaContextParams.rope_freq_scale);
  LlamaContextParams.yarn_ext_factor:=ReadJSONValue(SubObject,'yarn_ext_factor',LlamaContextParams.yarn_ext_factor);
  LlamaContextParams.yarn_attn_factor:=ReadJSONValue(SubObject,'yarn_attn_factor',LlamaContextParams.yarn_attn_factor);
  LlamaContextParams.yarn_beta_fast:=ReadJSONValue(SubObject,'yarn_beta_fast',LlamaContextParams.yarn_beta_fast);
  LlamaContextParams.yarn_beta_slow:=ReadJSONValue(SubObject,'yarn_beta_slow',LlamaContextParams.yarn_beta_slow);
  LlamaContextParams.yarn_orig_ctx:=ReadJSONValue(SubObject,'yarn_orig_ctx',LlamaContextParams.yarn_orig_ctx);
  LlamaContextParams.defrag_thold:=ReadJSONValue(SubObject,'defrag_thold',LlamaContextParams.defrag_thold);
  LlamaContextParams.type_k:=ReadJSONValue(SubObject,'type_k',LlamaContextParams.type_k);
  LlamaContextParams.type_v:=ReadJSONValue(SubObject,'type_v',LlamaContextParams.type_v);
  LlamaContextParams.embeddings:=ReadJSONValue(SubObject,'embeddings',LlamaContextParams.embeddings);
  LlamaContextParams.offload_kqv:=ReadJSONValue(SubObject,'offload_kqv',LlamaContextParams.offload_kqv);
  LlamaContextParams.no_perf:=ReadJSONValue(SubObject,'no_perf',LlamaContextParams.no_perf);
  LlamaContextParams.op_offload:=ReadJSONValue(SubObject,'op_offload',LlamaContextParams.op_offload);
  LlamaContextParams.swa_full:=ReadJSONValue(SubObject,'swa_full',LlamaContextParams.swa_full);
  LlamaContextParams.kv_unified:=ReadJSONValue(SubObject,'kv_unified',LlamaContextParams.kv_unified);
  LlamaContextParams.samplers:=pointer(ReadJSONValue(SubObject,'samplers',integer(LlamaContextParams.samplers)));
  LlamaContextParams.n_samplers:=ReadJSONValue(SubObject,'n_samplers',integer(LlamaContextParams.n_samplers));

  SubObject:=ReadJSONObject(AJSON, 'CompletationParams');
  CompletationParams.MinGenTokens:=ReadJSONValue(SubObject,'MinGenTokens',CompletationParams.MinGenTokens);
  CompletationParams.MaxGenTokens:=ReadJSONValue(SubObject,'MaxGenTokens',CompletationParams.MaxGenTokens);
  CompletationParams.Temperature:=ReadJSONValue(SubObject,'Temperature',CompletationParams.Temperature);
  CompletationParams.TopK:=ReadJSONValue(SubObject,'TopK',CompletationParams.TopK);
  CompletationParams.TopP:=ReadJSONValue(SubObject,'TopP',CompletationParams.TopP);
  CompletationParams.RepeatPenalty:=ReadJSONValue(SubObject,'RepeatPenalty',CompletationParams.RepeatPenalty);
  CompletationParams.RepeatLastN:=ReadJSONValue(SubObject,'RepeatLastN',CompletationParams.RepeatLastN);
end;

function TLlamaSettings.ToJSON: TJSONObject;
var
  SubObject: TJSONObject;
begin
  Result:=TJSONObject.Create;

  SubObject:=WriteJSONValue(Result,'LlamaModelParams',TJSONObject.Create);
  WriteJSONValue(SubObject,'devices',integer(LlamaModelParams.devices));
  WriteJSONValue(SubObject,'tensor_buft_overrides',integer(LlamaModelParams.tensor_buft_overrides));
  WriteJSONValue(SubObject,'n_gpu_layers',LlamaModelParams.n_gpu_layers);
  WriteJSONValue(SubObject,'split_mode',LlamaModelParams.split_mode);
  WriteJSONValue(SubObject,'main_gpu',LlamaModelParams.main_gpu);
  WriteJSONValue(SubObject,'tensor_split',integer(LlamaModelParams.tensor_split));
  WriteJSONValue(SubObject,'llama_model_kv_override',integer(LlamaModelParams.llama_model_kv_override));
  WriteJSONValue(SubObject,'vocab_only',LlamaModelParams.vocab_only);
  WriteJSONValue(SubObject,'use_mmap',LlamaModelParams.use_mmap);
  WriteJSONValue(SubObject,'use_direct_io',LlamaModelParams.use_direct_io);
  WriteJSONValue(SubObject,'use_mlock',LlamaModelParams.use_mlock);
  WriteJSONValue(SubObject,'check_tensors',LlamaModelParams.check_tensors);
  WriteJSONValue(SubObject,'use_extra_bufts',LlamaModelParams.use_extra_bufts);
  WriteJSONValue(SubObject,'no_host',LlamaModelParams.no_host);
  WriteJSONValue(SubObject,'no_alloc',LlamaModelParams.no_alloc);

  SubObject:=WriteJSONValue(Result,'LlamaContextParams',TJSONObject.Create);
  WriteJSONValue(SubObject,'n_ctx',LlamaContextParams.n_ctx);
  WriteJSONValue(SubObject,'n_batch',LlamaContextParams.n_batch);
  WriteJSONValue(SubObject,'n_ubatch',LlamaContextParams.n_ubatch);
  WriteJSONValue(SubObject,'n_seq_max',LlamaContextParams.n_seq_max);
  WriteJSONValue(SubObject,'n_threads',LlamaContextParams.n_threads);
  WriteJSONValue(SubObject,'n_threads_batch',LlamaContextParams.n_threads_batch);
  WriteJSONValue(SubObject,'rope_scaling_type',LlamaContextParams.rope_scaling_type);
  WriteJSONValue(SubObject,'pooling_type',LlamaContextParams.pooling_type);
  WriteJSONValue(SubObject,'llama_attention_type',LlamaContextParams.llama_attention_type);
  WriteJSONValue(SubObject,'llama_flash_attn_type',LlamaContextParams.llama_flash_attn_type);
  WriteJSONValue(SubObject,'rope_freq_base',LlamaContextParams.rope_freq_base);
  WriteJSONValue(SubObject,'rope_freq_scale',LlamaContextParams.rope_freq_scale);
  WriteJSONValue(SubObject,'yarn_ext_factor',LlamaContextParams.yarn_ext_factor);
  WriteJSONValue(SubObject,'yarn_attn_factor',LlamaContextParams.yarn_attn_factor);
  WriteJSONValue(SubObject,'yarn_beta_fast',LlamaContextParams.yarn_beta_fast);
  WriteJSONValue(SubObject,'yarn_beta_slow',LlamaContextParams.yarn_beta_slow);
  WriteJSONValue(SubObject,'yarn_orig_ctx',LlamaContextParams.yarn_orig_ctx);
  WriteJSONValue(SubObject,'defrag_thold',LlamaContextParams.defrag_thold);
  WriteJSONValue(SubObject,'type_k',LlamaContextParams.type_k);
  WriteJSONValue(SubObject,'type_v',LlamaContextParams.type_v);
  WriteJSONValue(SubObject,'embeddings',LlamaContextParams.embeddings);
  WriteJSONValue(SubObject,'offload_kqv',LlamaContextParams.offload_kqv);
  WriteJSONValue(SubObject,'no_perf',LlamaContextParams.no_perf);
  WriteJSONValue(SubObject,'op_offload',LlamaContextParams.op_offload);
  WriteJSONValue(SubObject,'swa_full',LlamaContextParams.swa_full);
  WriteJSONValue(SubObject,'kv_unified',LlamaContextParams.kv_unified);
  WriteJSONValue(SubObject,'samplers',integer(LlamaContextParams.samplers));
  WriteJSONValue(SubObject,'n_samplers',LlamaContextParams.n_samplers);

  SubObject:=WriteJSONValue(Result, 'CompletationParams', TJSONObject.Create);
  WriteJSONValue(SubObject,'MinGenTokens',CompletationParams.MinGenTokens);
  WriteJSONValue(SubObject,'MaxGenTokens',CompletationParams.MaxGenTokens);
  WriteJSONValue(SubObject,'Temperature',CompletationParams.Temperature);
  WriteJSONValue(SubObject,'TopK',CompletationParams.TopK);
  WriteJSONValue(SubObject,'TopP',CompletationParams.TopP);
  WriteJSONValue(SubObject,'RepeatPenalty',CompletationParams.RepeatPenalty);
  WriteJSONValue(SubObject,'RepeatLastN',CompletationParams.RepeatLastN);
end;

{ TTaskLoadModel }

constructor TTaskLoadModel.Create(
  AModelName: string;
  ALlamaSettings: TLlamaSettings
);
begin
  inherited Create(true);
  FreeOnTerminate:=true;
  FModelName:=AModelName;
  FLlamaSettings:=ALlamaSettings;
  Resume;
end;

procedure TTaskLoadModel.Execute;
begin
  try
    ModelInfo:=LoadModel(
      FModelName,
      FLlamaSettings.LlamaModelParams,
      FLlamaSettings.LlamaContextParams
    );
  finally
    Task:=nil;
  end;
end;

{ TTaskCompletation }

constructor TTaskCompletation.Create(
  AModelInfo: TModelInfo;
  ACompletationParams: TCompletationParams;
  APrompt: string)
;
begin
  inherited Create(true);
  FreeOnTerminate:=true;
  FModelInfo:=AModelInfo;
  FCompletationParams:=ACompletationParams;
  FPrompt:=APrompt;
  FState:=Default(TChatState);
  Resume;
end;

procedure TTaskCompletation.Execute;
begin
  try
    TaskResult:=Completion(
      FModelInfo,
      FState,
      FPrompt,
      FCompletationParams,
      TokenCallback
    );
  finally
    ModelInfo:=FModelInfo;
    Task:=nil;
  end;
end;

{ TTaskChatCompletation }

constructor TTaskChatCompletation.Create(
  AModelInfo: TModelInfo;
  ACompletationParams: TCompletationParams;
  AChatMessageArray: TArray<TChatMessage>
);
begin
  inherited Create(true);
  FreeOnTerminate:=true;
  FModelInfo:=AModelInfo;
  FCompletationParams:=ACompletationParams;
  FChatMessageArray:=AChatMessageArray;
  FState:=Default(TChatState);
  Resume;
end;

procedure TTaskChatCompletation.Execute;
begin
  try
    TaskResult:=ChatCompletion(
      FModelInfo,
      FState,
      FChatMessageArray,
      FCompletationParams,
      TokenCallback
    );
  finally
    ModelInfo:=FModelInfo;
    Task:=nil;
  end;
end;

exports
  LibInit,
  LibFree;

begin
end.
