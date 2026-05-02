unit uLLamaCPPLibBase;

interface

uses
  uLlamaCPPBinding,
  uUtility;

type
  TTemplateMode = (tmNative, tmFallback);

  TModelInfo = record
    Model: TLlamaInstance.llama_model;
    Context: TLlamaInstance.llama_context;
    Vocab: TLlamaInstance.llama_vocab;
    TemplateMode: TTemplateMode;
    RawTemplate: string;
    ContextParams: TLlamaInstance.llama_context_params;
  end;

function LoadModelParams: TLlamaInstance.llama_model_params;
function LoadContextParams: TLlamaInstance.llama_context_params;

function LoadModel(
  const ModelPath: string
): TModelInfo; overload;
function LoadModel(
  const ModelPath: string;
  const ModelParams: TLlamaInstance.llama_model_params;
  const ContextParams: TLlamaInstance.llama_context_params
): TModelInfo; overload;

procedure UnloadModel(
  var ModelInfo: TModelInfo
);

function HasChatTemplate(
  ModelInfo: TModelInfo;
  out ChatTemplate: string
): Boolean;

function HashString(const S: string): Cardinal;

function Tokenize(
  ModelInfo: TModelInfo; 
  const Text: string;
  AddSpecial: Boolean = True; 
  ParseSpecial: Boolean = True
): TArray<Integer>;

function CountTokens(
  ModelInfo: TModelInfo; 
  const Text: string
): Integer;

function TokenToString(const ModelInfo: TModelInfo; Token: Int32; IncludeSpecial: Boolean = true): string;

function TokensToString(const ModelInfo: TModelInfo; Tokens: TArray<Int32>; IncludeSpecial: Boolean = true): string;

implementation

function LoadModelParams: TLlamaInstance.llama_model_params;
begin
  Result:=LlamaInstance.llama_model_default_params;
end;

function LoadContextParams: TLlamaInstance.llama_context_params;
begin
  Result:=LlamaInstance.llama_context_default_params;
end;

function LoadModel(
  const ModelPath: string
): TModelInfo;
begin
  Result:=LoadModel(ModelPath, LoadModelParams, LoadContextParams);
end;

function LoadModel(
  const ModelPath: string;
  const ModelParams: TLlamaInstance.llama_model_params;
  const ContextParams: TLlamaInstance.llama_context_params
): TModelInfo;
var
  MetaValue: string;
begin
  Result.Model := LlamaInstance.llama_model_load_from_file(
    PAnsiChar(AnsiString(ModelPath)), 
    ModelParams
  );

  if (not Assigned(Result.Model)) then
    RaiseException('Load model error');

  Result.Context := LlamaInstance.llama_init_from_model(
    Result.Model,
    ContextParams
  );

  if (not Assigned(Result.Context)) then
    RaiseException('Init context error');

  Result.ContextParams:=ContextParams;

  Result.Vocab := LlamaInstance.llama_model_get_vocab(Result.Model);

  if (not Assigned(Result.Vocab)) then
    RaiseException('Getting vocab error');

  if (HasChatTemplate(Result, MetaValue)) then
    Result.RawTemplate := MetaValue
  else
    Result.RawTemplate := '';
end;

procedure UnloadModel(var ModelInfo: TModelInfo);
begin
  LlamaInstance.AddLogCallback(nil,nil);
  if (Assigned(ModelInfo.Context)) then
    try
      LlamaInstance.llama_free(ModelInfo.Context);
    except
    end;
  if (Assigned(ModelInfo.Model)) then
    try
      LlamaInstance.llama_free_model(ModelInfo.Model);
    except
    end;
  ModelInfo:=Default(TModelInfo);
end;

function HasChatTemplate(
  ModelInfo: TModelInfo;
  out ChatTemplate: string
): Boolean;
var
  Buffer: PAnsiChar;
  BufSize, Len: Integer;
  Utf8: UTF8String;
begin
  BufSize := 32768;
  GetMem(Buffer, BufSize);
  try
    Len:=LlamaInstance.llama_model_meta_val_str(
      ModelInfo.Model,
      'tokenizer.chat_template', 
      Buffer, 
      BufSize
    );
    if (Len<=0) then
      begin
        ChatTemplate := '';
        Exit(False);
      end;
    SetString(Utf8, PAnsiChar(@Buffer[0]), Len);
    ChatTemplate:=UTF8ToString(Utf8);
    Result:= True;
  finally
    FreeMem(Buffer);
  end;
end;

function HashString(const S: string): Cardinal;
var
  i: Integer;
begin
  Result := 1469598103934665603; 

  for i := 1 to Length(S) do
    begin
      Result := Result xor Ord(S[i]);
      Result := Result * 1099511628211;
    end;
end;

function Tokenize(
  ModelInfo: TModelInfo; 
  const Text: string;
  AddSpecial: Boolean; 
  ParseSpecial: Boolean
): TArray<Integer>;
var
  Utf8: UTF8String;
  Count: Integer;
  Buffer: array of Int32;
begin
  Utf8 := UTF8Encode(Text);

  if Length(Utf8) = 0 then
    Result := [];

  SetLength(Buffer, Length(Utf8) + 32);

  Count := LlamaInstance.llama_tokenize(
    ModelInfo.Vocab, 
    PAnsiChar(Utf8),
    Length(Utf8), 
    @Buffer[0], 
    Length(Buffer), 
    AddSpecial, 
    ParseSpecial
  );

  if Count < 0 then
    RaiseException('Errore tokenize (data): %d', [Count]);

  SetLength(Buffer, Count);
  Result := TArray<Integer>(Buffer);
end;

function CountTokens(
  ModelInfo: TModelInfo; 
  const Text: string
): Integer;
var
  Tokens: TArray<Int32>;
begin
  Tokens := Tokenize(ModelInfo, PAnsiChar(AnsiString(Text)));
  Result := Length(Tokens);
end;

function TokenToString(const ModelInfo: TModelInfo; Token: Int32; IncludeSpecial: Boolean): string;
var
  Buffer: PAnsiChar;
  BufSize: Integer;
  Len: Integer;
  Utf8: UTF8String;
begin
  BufSize := 32768;
  GetMem(Buffer, BufSize);
  try
    Len := LlamaInstance.llama_token_to_piece(
      ModelInfo.Vocab,
      Token,
      Buffer,
      BufSize,
      0,
      IncludeSpecial
    );

    if Len < 0 then
      Exit('');

    SetString(Utf8, PAnsiChar(@Buffer[0]), Len);
    Result := UTF8ToString(Utf8);
  finally
    FreeMem(Buffer);
  end;
end;

function TokensToString(const ModelInfo: TModelInfo; Tokens: TArray<Int32>; IncludeSpecial: Boolean): string;
var
  i: Integer;
begin
  Result := '';
  for i := 0 to High(Tokens) do
    Result := Result + TokenToString(ModelInfo, Tokens[i], IncludeSpecial);
end;

end.
