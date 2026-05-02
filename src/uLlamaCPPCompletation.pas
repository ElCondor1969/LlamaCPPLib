unit uLlamaCPPCompletation;

interface

uses
  System.Classes,
  System.SysUtils,
  System.Math,
  System.Generics.Defaults,
  System.Generics.Collections,
  uLlamaCPPBinding,
  uLLamaCPPLibBase;

type
  TChatRole = (crSystem, crUser, crAssistant, crTool);

  TChatMessage = record
  public
    Role: TChatRole;
    Content: string;
    // cache token
    Tokens: TArray<Integer>;
    IsTokenized: Boolean;
    constructor Create(ARole: TChatRole; AContent: string);
  end;

  TChatState = record
    Tokens: TArray<Int32>;
    n_past: Integer;
    // cache prompt
    LastPromptHash: UInt64;
    Prefill: Boolean;
  end;

  TCompletationParams = record
    MinGenTokens: Integer;
    MaxGenTokens: Integer;
    Temperature: Single;
    TopK: Integer;
    TopP: Single;
    RepeatPenalty: Single;
    RepeatLastN: Integer;
  end;

  TOnTokenGenerated = reference to procedure(
    const Token: Integer; 
    const TokenText: string;
    var FlagContinue: Boolean
  );

function ChatCompletion(
  var ModelInfo: TModelInfo;
  var State: TChatState;
  var Messages: TArray<TChatMessage>;
  const Params: TCompletationParams;
  const OnToken: TOnTokenGenerated = nil
): string;

function Completion(
  var ModelInfo: TModelInfo;
  var State: TChatState;
  const Prompt: string;
  const Params: TCompletationParams;
  const OnToken: TOnTokenGenerated = nil
): string;

implementation

procedure ResetContext(
  var ModelInfo: TModelInfo; 
  var State: TChatState
);
var 
  ContextParams: TLlamaInstance.llama_context_params;
begin
  LlamaInstance.llama_free(ModelInfo.Context);

  ModelInfo.Context := LlamaInstance.llama_init_from_model(
    ModelInfo.Model,
    ModelInfo.ContextParams
  );

  State.Tokens := [];
  State.n_past := 0;
end;

function CreateBatch(
  const Tokens: TArray<Int32>; StartPos: Integer
): TLlamaInstance.llama_batch;
var
  i: Integer;
  N: Integer;
begin
  N := Length(Tokens);

  if N = 0 then
    raise Exception.Create('CreateBatch: Tokens vuoto');

  Result := LlamaInstance.llama_batch_init(N, 0, 1);

  if (Result.Token = nil) or (Result.pos = nil) or (Result.n_seq_id = nil) or
    (Result.seq_id = nil) or (Result.logits = nil) then
    raise Exception.Create('CreateBatch: batch non valido (binding errato?)');

  for i := 0 to N - 1 do
  begin
    Result.Token[i] := Tokens[i];
    Result.pos[i] := StartPos + i;
    Result.n_seq_id[i] := 1;
    Result.seq_id[i][0] := 0;
    Result.logits[i] := 0;
  end;

  // logits only for last token
  Result.logits[N - 1] := 1;
  Result.n_tokens := N;
end;

procedure EnsureMessageTokenized(
  var Msg: TChatMessage;
  const ModelInfo: TModelInfo
);
begin
  if Msg.IsTokenized then
    Exit;

  Msg.Tokens := Tokenize(
    ModelInfo,
    Msg.Content,
    False, // No BOS for single messages
    True
  );

  Msg.IsTokenized := True;
end;

procedure EnsureAllMessagesTokenized(
  var Messages: TArray<TChatMessage>;
  const ModelInfo: TModelInfo
);
var
  i: Integer;
begin
  for i := 0 to High(Messages) do
    EnsureMessageTokenized(Messages[i], ModelInfo);
end;

function TestChatTemplate: Boolean;
var
  Msg: array [0 .. 0] of TLlamaInstance.llama_chat_message;
  Role, Content: AnsiString;
  Needed: Integer;
begin
  Role := 'user';
  Content := 'test';

  Msg[0].Role := PAnsiChar(Role);
  Msg[0].Content := PAnsiChar(Content);

  Needed := LlamaInstance.llama_chat_apply_template(
    nil, 
    @Msg[0], 
    1,
    True, 
    nil, 
    0
  );

  Result:=(Needed > 0);
end;

procedure InitializeTemplate(var ModelInfo: TModelInfo);
var
  Meta: string;
begin
  ModelInfo.TemplateMode := tmFallback;
  HasChatTemplate(ModelInfo, ModelInfo.RawTemplate);

  if not TestChatTemplate then
    ModelInfo.TemplateMode := tmNative;
end;

function ApplyFallbackTemplate(
  const Messages: TArray<TChatMessage>
): string;
var
  Msg: TChatMessage;
begin
  Result := '';

  for Msg in Messages do
    case Msg.Role of
      crSystem:
        Result := Result + 'System: ' + Msg.Content + sLineBreak;

      crUser:
        Result := Result + 'User: ' + Msg.Content + sLineBreak;

      crAssistant:
        Result := Result + 'Assistant: ' + Msg.Content + sLineBreak;
    end;

  Result := Result + 'Assistant: ';
end;

function ApplyChatTemplateNative(
  Model: TLlamaInstance.llama_model;
  const Messages: TArray<TChatMessage>
): string;
var
  MsgArray: array of TLlamaInstance.llama_chat_message;
  Roles: array of AnsiString;
  Contents: array of AnsiString;
  i: Integer;
  Needed: Integer;
  Buffer: PAnsiChar;
  Utf8: UTF8String;
begin
  SetLength(MsgArray, Length(Messages));
  SetLength(Roles, Length(Messages));
  SetLength(Contents, Length(Messages));

  for i := 0 to High(Messages) do
    begin
      case Messages[i].Role of
        crSystem:
          Roles[i] := 'system';
        crUser:
          Roles[i] := 'user';
        crAssistant:
          Roles[i] := 'assistant';
      end;

      Contents[i] := AnsiString(Messages[i].Content);

      MsgArray[i].Role := PAnsiChar(Roles[i]);
      MsgArray[i].Content := PAnsiChar(Contents[i]);
    end;

  // 1. Get needed dimension
  Needed := LlamaInstance.llama_chat_apply_template(
    nil, // Use model template
    @MsgArray[0],
    Length(MsgArray),
    True,
    nil,
    0
  );

  if Needed <= 0 then
    raise Exception.Create('Errore nel template');

  // 2. Allocate buffer
  GetMem(Buffer, Needed);

  try
    // 3. Generate prompt
    LlamaInstance.llama_chat_apply_template(
      nil,
      @MsgArray[0],
      Length(MsgArray),
      True,
      Buffer,
      Needed
    );

    SetString(Utf8, PAnsiChar(@Buffer[0]), Needed);
    Result := UTF8ToString(Utf8);
  finally
    FreeMem(Buffer);
  end;
end;

function ApplyChatTemplateSafe(const ModelInfo: TModelInfo;
  const Messages: TArray<TChatMessage>): string;
begin
  case ModelInfo.TemplateMode of
    tmNative:
      begin
        try
          Result := ApplyChatTemplateNative(ModelInfo.Model, Messages);

          if Result.Trim = '' then
            raise Exception.Create('Template vuoto');

        except
          // fallback
          Result := ApplyFallbackTemplate(Messages);
        end;
      end;

    tmFallback:
      Result := ApplyFallbackTemplate(Messages);
  end;
end;

function BuildFittedPrompt(
  const ModelInfo: TModelInfo;
  const Messages: TArray<TChatMessage>;
  Params: TCompletationParams
): string;
var
  ctxSize: Integer;
  Budget: Integer;
  i: Integer;
  TempMessages: TArray<TChatMessage>;
  SystemMsg: TChatMessage;
  HasSystem: Boolean;
  Prompt: string;

  function FitsBudget(const Msgs: TArray<TChatMessage>): Boolean;
  begin
    Prompt := ApplyChatTemplateSafe(ModelInfo, Msgs);
    Result := CountTokens(ModelInfo, Prompt) <= Budget;
  end;

begin
  ctxSize := LlamaInstance.llama_n_ctx(ModelInfo.Context);

  // Space reserved for generation
  Budget := Max(ctxSize - Params.MaxGenTokens,ctxSize - Params.MinGenTokens);

  TempMessages := [];
  HasSystem := False;

  // 1. extract SYSTEM (only once)
  for var Msg in Messages do
    begin
      if Msg.Role = crSystem then
      begin
        SystemMsg := Msg;
        HasSystem := True;
        Break;
      end;
    end;

  // 2. ALWAYS start from the last message (usually USER)
  i := High(Messages);

  while i >= 0 do
    begin
      if Messages[i].Role = crSystem then
        begin
          Dec(i);
          Continue;
        end;

      // Logic block: user + assistant
      if (Messages[i].Role = crAssistant) and (i > 0) and (Messages[i-1].Role = crUser) then
        begin
          // try adding the pair
          var Candidate := [Messages[i-1], Messages[i]] + TempMessages;

          if HasSystem then
            Candidate := [SystemMsg] + Candidate;

          if FitsBudget(Candidate) then
            begin
              TempMessages := [Messages[i-1], Messages[i]] + TempMessages;
              Dec(i, 2);
              Continue;
            end
          else
            Break;
        end
      else
        begin
          // single message (typically last USER)
          var Candidate := [Messages[i]] + TempMessages;

          if HasSystem then
            Candidate := [SystemMsg] + Candidate;

          if FitsBudget(Candidate) then
            begin
              TempMessages := [Messages[i]] + TempMessages;
              Dec(i);
              Continue;
            end
          else
            Break;
        end;
    end;

  // 3. prepend SYSTEM at the end (guaranteed in the head)
  if HasSystem then
    TempMessages := [SystemMsg] + TempMessages;

  // 4. final prompt
  Result := ApplyChatTemplateSafe(ModelInfo, TempMessages);
end;

procedure PrefillContext(
  ModelInfo: TModelInfo;
  const Tokens: TArray<Int32>;
  var State: TChatState);
var
  Batch: TLlamaInstance.llama_batch;
  ReturnValue: Integer;
begin
  // Prefill ALWAYS starts from position 0
  Batch := CreateBatch(Tokens, 0);

  ReturnValue := LlamaInstance.llama_decode(ModelInfo.Context, Batch);
  if ReturnValue <> 0 then
    raise Exception.CreateFmt('Errore decode %d (prefill)', [ReturnValue]);

  LlamaInstance.llama_batch_free(Batch);

  State.Tokens := Tokens;
  State.n_past := Length(Tokens);
  State.Prefill := True;
end;

procedure AppendToContext(ctx: TLlamaInstance.llama_context;
  const Tokens: TArray<Int32>; var State: TChatState);
var
  Batch: TLlamaInstance.llama_batch;
  ReturnValue: Integer;
begin
  if Length(Tokens) = 0 then
    Exit;

  Batch := CreateBatch(Tokens, State.n_past);

  ReturnValue := LlamaInstance.llama_decode(ctx, Batch);
  if ReturnValue <> 0 then
    raise Exception.CreateFmt('Errore decode %d (append)', [ReturnValue]);

  LlamaInstance.llama_batch_free(Batch);

  State.Tokens := State.Tokens + Tokens;
  Inc(State.n_past, Length(Tokens));
end;

function GetSpecialTokenId(
  ModelInfo: TModelInfo;
  const TokenStr: string
): Integer;
var
  Toks: TArray<Integer>;
begin
  Toks := Tokenize(ModelInfo, TokenStr, False, True);

  if Length(Toks) = 0 then
    Result := -1
  else
    Result := Toks[0];
end;

function ExtractStopSequences(
  const ModelInfo: TModelInfo
): TArray<integer>;
begin
  Result:=[
    LlamaInstance.llama_token_eot(ModelInfo.Vocab),
    GetSpecialTokenId(ModelInfo,'<|eot_id|>'),
    GetSpecialTokenId(ModelInfo,'<|end_of_text|>'),
    GetSpecialTokenId(ModelInfo,'<|start_header_id|>'),
    GetSpecialTokenId(ModelInfo,'<|fim_im_end|>'),
    GetSpecialTokenId(ModelInfo,'<|im_end|>'),
    GetSpecialTokenId(ModelInfo,'<|user|>'),
    GetSpecialTokenId(ModelInfo,'[INST]'),
    GetSpecialTokenId(ModelInfo,'</s>')
  ];
end;

procedure ApplyRepetitionPenaltyFast(var Candidates
  : TLlamaInstance.llama_token_data_array; const LastTokens: TArray<Int32>;
  Penalty: Single);
var
  Seen: TDictionary<Integer, Boolean>;
  i: Integer;
begin
  if (Penalty = 1.0) or (Length(LastTokens) = 0) then
    Exit;

  Seen := TDictionary<Integer, Boolean>.Create;
  try
    for var T in LastTokens do
      Seen.AddOrSetValue(T, True);

    for i := 0 to Candidates.size - 1 do
    begin
      if Seen.ContainsKey(Candidates.data[i].id) then
      begin
        if Candidates.data[i].logit < 0 then
          Candidates.data[i].logit := Candidates.data[i].logit * Penalty
        else
          Candidates.data[i].logit := Candidates.data[i].logit / Penalty;
      end;
    end;

  finally
    Seen.Free;
  end;
end;

procedure ApplyTemperature(var Candidates
  : TLlamaInstance.llama_token_data_array; Temperature: Single);
var
  i: Integer;
begin
  if (Temperature <= 0) or (Temperature = 1.0) then
    Exit;

  for i := 0 to Candidates.size - 1 do
    Candidates.data[i].logit := Candidates.data[i].logit / Temperature;
end;

procedure ApplySoftmax(var Candidates: TLlamaInstance.llama_token_data_array);
var
  i: Integer;
  MaxLogit, Sum: Double;
begin
  // find max for numerical stability
  MaxLogit := Candidates.data[0].logit;
  for i := 1 to Candidates.size - 1 do
    if Candidates.data[i].logit > MaxLogit then
      MaxLogit := Candidates.data[i].logit;

  // normalized exp
  Sum := 0.0;
  for i := 0 to Candidates.size - 1 do
  begin
    Candidates.data[i].p := Exp(Candidates.data[i].logit - MaxLogit);
    Sum := Sum + Candidates.data[i].p;
  end;

  // normalization
  for i := 0 to Candidates.size - 1 do
    Candidates.data[i].p := Candidates.data[i].p / Sum;
end;

procedure ApplyTopK(var Candidates: TLlamaInstance.llama_token_data_array;
  K: Integer);
var
  Arr: TArray<TLlamaInstance.llama_token_data>;
begin
  if (K <= 0) or (K >= Candidates.size) then
    Exit;

  // create array view on C memory
  SetLength(Arr, Candidates.size);
  Move(Candidates.data^[0], Arr[0], Candidates.size *
    SizeOf(TLlamaInstance.llama_token_data));

  // sort
  TArray.Sort<TLlamaInstance.llama_token_data>(Arr,
    TComparer<TLlamaInstance.llama_token_data>.Construct(
    function(const L, R: TLlamaInstance.llama_token_data): Integer
    begin
      Result := CompareValue(R.p, L.p);
    end));

  // back copy
  Move(Arr[0], Candidates.data^[0], Candidates.size *
    SizeOf(TLlamaInstance.llama_token_data));

  Candidates.size := K;
end;

procedure ApplyTopP(var Candidates: TLlamaInstance.llama_token_data_array;
p: Double);
var
  i: Integer;
  CumProb: Double;
  Arr: TArray<TLlamaInstance.llama_token_data>;
begin
  if p >= 1.0 then
    Exit;

  if Candidates.size <= 1 then
    Exit;

  SetLength(Arr, Candidates.size);
  Move(Candidates.data^[0], Arr[0], Candidates.size *
    SizeOf(TLlamaInstance.llama_token_data));

  // sort by decreasing probability
  TArray.Sort<TLlamaInstance.llama_token_data>(Arr,
    TComparer<TLlamaInstance.llama_token_data>.Construct(
    function(const L, R: TLlamaInstance.llama_token_data): Integer
    begin
      Result := CompareValue(R.p, L.p);
    end));

  // cumulative
  CumProb := 0.0;

  for i := 0 to High(Arr) do
  begin
    CumProb := CumProb + Arr[i].p;

    if CumProb >= p then
    begin
      Candidates.size := i + 1;
      Break;
    end;
  end;

  // copy back (only the first Candidates.size elements)
  Move(Arr[0], Candidates.data^[0], Candidates.size *
    SizeOf(TLlamaInstance.llama_token_data));
end;

function SampleFromDistribution(const Candidates
  : TLlamaInstance.llama_token_data_array): Integer;
var
  R, Cum: Double;
  i: Integer;
begin
  R := Random; // 0..1
  Cum := 0.0;

  for i := 0 to Candidates.size - 1 do
  begin
    Cum := Cum + Candidates.data[i].p;

    if R <= Cum then
      Exit(Candidates.data[i].id);
  end;

  Result := Candidates.data[Candidates.size - 1].id;
end;

function SampleTokenAdvanced(
  const ModelInfo: TModelInfo;
  const Params: TCompletationParams; 
  var State: TChatState
): Integer;
var
  logits: TLlamaInstance.llama_logit_array;
  Candidates: TLlamaInstance.llama_token_data_array;
  VocabSize: Integer;
  i: Integer;
  Arr: TLlamaInstance.PTokenDataArray;
begin
  // 1. logits
  if (State.Prefill) then
  begin
    i := State.n_past - 1;
    State.Prefill := False;
  end
  else
    i := 0;
  logits := LlamaInstance.llama_get_logits_ith(ModelInfo.Context, i);

  VocabSize := LlamaInstance.llama_vocab_n_tokens(ModelInfo.Vocab);

  GetMem(Candidates.data, VocabSize * SizeOf(TLlamaInstance.llama_token_data));
  Arr := Candidates.data;

  for i := 0 to VocabSize - 1 do
  begin
    Arr[i].id := i;
    Arr[i].logit := logits[i];
    Arr[i].p := 0.0;
  end;

  Candidates.size := VocabSize;
  Candidates.sorted := False;

  try
    // 2. repetition
    ApplyRepetitionPenaltyFast(Candidates, State.Tokens, Params.RepeatPenalty);

    // 3. temperature
    ApplyTemperature(Candidates, Params.Temperature);

    // 4. softmax
    ApplySoftmax(Candidates);

    // 5. top-k
    ApplyTopK(Candidates, Params.TopK);

    // 6. top-p
    ApplyTopP(Candidates, Params.TopP);

    // 7. sample
    Result := SampleFromDistribution(Candidates);

  finally
    FreeMem(Candidates.data);
  end;
end;

function GenerateWithAdvancedSampling(
  const ModelInfo: TModelInfo;
  var State: TChatState;
  const Params: TCompletationParams;
  const StopSeq: TArray<integer>;
  const OnToken: TOnTokenGenerated
): string;
var
  i, Token: Integer;
  Batch: TLlamaInstance.llama_batch;
  Piece: string;
  FlagContinue: Boolean;
  ReturnValue: Integer;
begin
  Result := '';
  FlagContinue := True;

  for i := 0 to Params.MaxGenTokens - 1 do
    begin
      // 1. sample
      Token := SampleTokenAdvanced(ModelInfo, Params, State);

      // 2. stop sequences
      for var Stop in StopSeq do
        if Stop=Token then
          Exit;

      // 3. append token al contesto
      Batch:=CreateBatch([Token], State.n_past);

      try
        ReturnValue := LlamaInstance.llama_decode(ModelInfo.Context, Batch);
        if ReturnValue <> 0 then
          Break;
      finally
        LlamaInstance.llama_batch_free(Batch);
      end;

      Inc(State.n_past);
      State.Tokens := State.Tokens + [Token];

      // 4. token conversion
      Piece := TokenToString(ModelInfo, Token, false);

      if Assigned(OnToken) then
        OnToken(Token, Piece, FlagContinue);

      Result := Result + Piece;

      if (not FlagContinue) then
        Break;
    end;
end;

function TokensDiffer(const A, B: TArray<Integer>): Boolean;
var
  i, MinLen: Integer;
begin
  MinLen := Min(Length(A), Length(B));

  for i := 0 to MinLen - 1 do
  begin
    if A[i] <> B[i] then
      Exit(True);
  end;

  // if different lengths then they differ
  Result := Length(A) <> Length(B);
end;

function CommonPrefixLength(const A, B: TArray<Integer>): Integer;
var
  i, MinLen: Integer;
begin
  MinLen := Min(Length(A), Length(B));

  for i := 0 to MinLen - 1 do
  begin
    if A[i] <> B[i] then
      Exit(i);
  end;

  Result := MinLen;
end;

function ChatCompletion(
  var ModelInfo: TModelInfo; 
  var State: TChatState;
  var Messages: TArray<TChatMessage>; 
  const Params: TCompletationParams;
  const OnToken: TOnTokenGenerated
): string;
var
  Prompt: string;
  Tokens: TArray<Int32>;
  PromptHash: UInt64;
  PrefixLen: Integer;
  DeltaTokens: TArray<Int32>;
  StopSeq: TArray<integer>;
  NeedReset: Boolean;
begin
  State.Prefill := True;

  // 1. Message cache token
  EnsureAllMessagesTokenized(Messages, ModelInfo);

  // 2. Prompt construction (with context window mgmt included)
  Prompt := BuildFittedPrompt(ModelInfo, Messages, Params);

  // 3. Hash prompt (cache)
  PromptHash := HashString(Prompt);

  if PromptHash <> State.LastPromptHash then
    begin
      // 4. Tokenization ONLY ONCE
      Tokens := Tokenize(ModelInfo, Prompt, True, True);

      // 5. Context Management
      PrefixLen := CommonPrefixLength(State.Tokens, Tokens);

      NeedReset := (PrefixLen = 0) or (PrefixLen <> Length(State.Tokens));

      if NeedReset then
        begin
          // reset context
          ResetContext(ModelInfo, State);

          PrefillContext(ModelInfo, Tokens, State);
        end
      else
        begin
          // pure append
          DeltaTokens := Copy(Tokens, PrefixLen, Length(Tokens) - PrefixLen);
          AppendToContext(ModelInfo.Context, DeltaTokens, State);
        end;

      State.LastPromptHash := PromptHash;
    end;

  // 6. Generation
  StopSeq := ExtractStopSequences(ModelInfo);

  Result := GenerateWithAdvancedSampling(
    ModelInfo,
    State,
    Params,
    StopSeq,
    OnToken
  );
end;

function Completion(
  var ModelInfo: TModelInfo;
  var State: TChatState;
  const Prompt: string;
  const Params: TCompletationParams;
  const OnToken: TOnTokenGenerated
): string;
var
  Tokens: TArray<Int32>;
  PromptHash: UInt64;
  PrefixLen: Integer;
  DeltaTokens: TArray<Int32>;
  StopSeq: TArray<Integer>;
  NeedReset: Boolean;
begin
  State.Prefill := True;

  // 1. Hash prompt (cache)
  PromptHash := HashString(Prompt);

  if PromptHash <> State.LastPromptHash then
    begin
      // 2. Tokenization
      Tokens := Tokenize(ModelInfo, Prompt, True, True);

      // 3. Context reuse
      PrefixLen := CommonPrefixLength(State.Tokens, Tokens);

      NeedReset := (PrefixLen = 0) or (PrefixLen <> Length(State.Tokens));

      if NeedReset then
        begin
          ResetContext(ModelInfo, State);
          PrefillContext(ModelInfo, Tokens, State);
        end
      else
        begin
          DeltaTokens := Copy(Tokens, PrefixLen, Length(Tokens) - PrefixLen);
          AppendToContext(ModelInfo.Context, DeltaTokens, State);
        end;

      State.LastPromptHash := PromptHash;
    end;

  // 4. Stop sequences
  StopSeq := ExtractStopSequences(ModelInfo);

  // 5. Generation
  Result := GenerateWithAdvancedSampling(
    ModelInfo,
    State,
    Params,
    StopSeq,
    OnToken
  );
end;

{ TChatMessage }

constructor TChatMessage.Create(ARole: TChatRole; AContent: string);
begin
  Role := ARole;
  Content := AContent;
end;

initialization
  SetExceptionMask(GetExceptionMask + [exOverflow, exInvalidOp]);

end.
