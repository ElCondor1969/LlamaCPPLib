unit uLlamaCPPEmbeddings;

interface

uses
  System.Classes,
  System.SysUtils,
  System.Math,
  System.Generics.Defaults,
  System.Generics.Collections,
  uLlamaCPPBinding,
  uLLamaCPPLibBase;

function GetTokenEmbedding(
  ModelInfo: TModelInfo;
  Token: Int32
): TArray<Single>;

implementation

function GetTokenEmbedding(ModelInfo:TModelInfo;Token:Int32):TArray<Single>;
var
  VocabSize, EmbdSize, Dim: Integer;
  Batch: TLlamaInstance.llama_batch;
  EmbeddingPtr: TLlamaInstance.PSingle;
  MemObj: TLlamaInstance.llama_memory_t;
begin
  Result:=nil;
  try
    VocabSize := LlamaInstance.llama_vocab_n_tokens(ModelInfo.Vocab);
    EmbdSize := LlamaInstance.llama_model_n_embd(ModelInfo.Model);
    Batch := LlamaInstance.llama_batch_init(1, 0, 1);
    Batch.token[0] := Token;
    Batch.pos[0]   := 0;
    Batch.n_seq_id[0] := 1;
    Batch.seq_id[0][0] := 0;
    Batch.logits[0] := 0;
    Batch.n_tokens := 1;
    MemObj := LlamaInstance.llama_get_memory(ModelInfo.Context);
    if not Assigned(MemObj) then
      raise Exception.Create('KV memory not available');
    LlamaInstance.llama_memory_seq_rm(MemObj, -1, 0, -1);
    if LlamaInstance.llama_decode(ModelInfo.Context, Batch) <> 0 then
      raise Exception.Create('Token decoding error');
    EmbeddingPtr := LlamaInstance.llama_get_embeddings_ith(ModelInfo.Context, 0);
    if (not Assigned(EmbeddingPtr)) then
      raise Exception.Create('Getting embedding error');
    SetLength(Result, EmbdSize);
    for Dim := 0 to EmbdSize - 1 do
      begin
        Result[Dim] := EmbeddingPtr^;
        Inc(EmbeddingPtr,SizeOf(Single));
      end;
  finally
    LlamaInstance.llama_batch_free(Batch);
  end;
end;

end.
