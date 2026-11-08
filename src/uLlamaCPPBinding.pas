unit uLlamaCPPBinding;

interface

uses
  System.SysUtils,
  System.IOUtils,
{$IFDEF MSWINDOWS}
  Winapi.Windows,
{$ENDIF}
  uDynLibLoader;

type
{$MINENUMSIZE 4}
  TGGMLType = (GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q4_2 = 4, // support has been removed
    GGML_TYPE_Q4_3 = 5, // support has been removed
    GGML_TYPE_Q5_0 = 6, GGML_TYPE_Q5_1 = 7, GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9, GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13, GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15, GGML_TYPE_IQ2_XXS = 16, GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18, GGML_TYPE_IQ1_S = 19, GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21, GGML_TYPE_IQ2_S = 22, GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24, GGML_TYPE_I16 = 25, GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27, GGML_TYPE_F64 = 28, GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30, GGML_TYPE_Q4_0_4_4 = 31, GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33, GGML_TYPE_TQ1_0 = 34, GGML_TYPE_TQ2_0 = 35,
    GGML_TYPE_COUNT = Integer(36) // Number of types (excluding the commented ones)
  );

  TGgmlLogLevel = (
    GGML_LOG_LEVEL_NONE = 0,
    GGML_LOG_LEVEL_DEBUG = 1,
    GGML_LOG_LEVEL_INFO = 2,
    GGML_LOG_LEVEL_WARN = 3,
    GGML_LOG_LEVEL_ERROR = 4,
    GGML_LOG_LEVEL_CONT = 5
  );

  TLlamaVocabType = (
    LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
    LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV = 5  // RWKV tokenizer based on greedy tokenization
  );

  TLlamaVocabPreType = (
    LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,  // Default pre-tokenization type
    LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,  // Pre-tokenization for LLaMA3
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,  // DeepSeek LLM tokenizer
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,  // DeepSeek Coder tokenizer
    LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,  // Falcon tokenizer
    LLAMA_VOCAB_PRE_TYPE_MPT            = 5,  // MPT tokenizer
    LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,  // StarCoder tokenizer
    LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,  // GPT-2 tokenizer
    LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,  // ReFact tokenizer
    LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,  // Command-R tokenizer
    LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10, // StableLM v2 tokenizer
    LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11, // QWEN v2 tokenizer
    LLAMA_VOCAB_PRE_TYPE_OLMO           = 12, // Olmo tokenizer
    LLAMA_VOCAB_PRE_TYPE_DBRX           = 13, // DBRX tokenizer
    LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14, // Smaug tokenizer
    LLAMA_VOCAB_PRE_TYPE_PORO           = 15, // Poro tokenizer
    LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16, // ChatGLM v3 tokenizer
    LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17, // ChatGLM v4 tokenizer
    LLAMA_VOCAB_PRE_TYPE_VIKING         = 18, // Viking tokenizer
    LLAMA_VOCAB_PRE_TYPE_JAIS           = 19, // Jais tokenizer
    LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20, // Tekken tokenizer
    LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21, // Smollm tokenizer
    LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22, // CodeShell tokenizer
    LLAMA_VOCAB_PRE_TYPE_BLOOM          = 23, // Bloom tokenizer
    LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24, // GPT-3 Finnish tokenizer
    LLAMA_VOCAB_PRE_TYPE_EXAONE         = 25, // ExaOne tokenizer
    LLAMA_VOCAB_PRE_TYPE_CHAMELEON      = 26  // Chameleon tokenizer
  );

  TLlamaRopeType = (
    LLAMA_ROPE_TYPE_NONE = -1,  // No rope type
    LLAMA_ROPE_TYPE_NORM =  0,  // Normal rope type
    LLAMA_ROPE_TYPE_NEOX =  2   // Neox rope type (maps to an external constant or definition)
  );

  TLlamaTokenType = (
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0, // Undefined token type
    LLAMA_TOKEN_TYPE_NORMAL       = 1, // Normal token
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2, // Unknown token
    LLAMA_TOKEN_TYPE_CONTROL      = 3, // Control token
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4, // User-defined token
    LLAMA_TOKEN_TYPE_UNUSED       = 5, // Unused token
    LLAMA_TOKEN_TYPE_BYTE         = 6  // Byte token
  );

  TLlamaTokenAttr = (
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0,      // Undefined attribute
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 shl 0, // Unknown attribute
    LLAMA_TOKEN_ATTR_UNUSED       = 1 shl 1, // Unused attribute
    LLAMA_TOKEN_ATTR_NORMAL       = 1 shl 2, // Normal attribute
    LLAMA_TOKEN_ATTR_CONTROL      = 1 shl 3, // Control attribute (SPECIAL?)
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 shl 4, // User-defined attribute
    LLAMA_TOKEN_ATTR_BYTE         = 1 shl 5, // Byte attribute
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 shl 6, // Normalized attribute
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 shl 7, // Left strip attribute
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 shl 8, // Right strip attribute
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 shl 9  // Single word attribute
  );

  TLlamaFType = (
    LLAMA_FTYPE_ALL_F32              = 0,   // All tensors in F32
    LLAMA_FTYPE_MOSTLY_F16           = 1,   // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_0          = 2,   // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_1          = 3,   // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,   // tok_embeddings.weight and output.weight are F16
    LLAMA_FTYPE_MOSTLY_Q4_2          = 5,   // Support has been removed
    LLAMA_FTYPE_MOSTLY_Q4_3          = 6,   // Support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0          = 7,   // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q5_0          = 8,   // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q5_1          = 9,   // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q2_K          = 10,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q6_K          = 18,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ1_S         = 24,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ3_S         = 26,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ3_M         = 27,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ2_S         = 28,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ2_M         = 29,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_IQ1_M         = 31,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_BF16          = 32,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_TQ1_0         = 36,  // Except 1D tensors
    LLAMA_FTYPE_MOSTLY_TQ2_0         = 37,  // Except 1D tensors
    LLAMA_FTYPE_GUESSED              = 1024 // Not specified in the model file
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
    LLAMA_POOLING_TYPE_RANK = Integer(4)  // used by reranking models to attach the classification head to the graph
  );

  TLlamaAttentionType = (
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = Integer(1)
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

  TLlamaModelKvOverrideType = (
    LLAMA_KV_OVERRIDE_TYPE_INT   = 0,   // Integer type
    LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1,   // Float type
    LLAMA_KV_OVERRIDE_TYPE_BOOL  = 2,   // Boolean type
    LLAMA_KV_OVERRIDE_TYPE_STR   = 3    // String type
  );
{$MINENUMSIZE 1}

{$ALIGN 8}
  TLlamaInstance = record
  private
    FLibPath: string;
    FHandleLLamaLib: THandle;
    FHandleGgmlLib: THandle;
    function GetProcAddress(const AHandle: THandle; const AProcName: string): Pointer;
    function LoadLib(const LibName: string): THandle;
    procedure LoadLibProcs;
  private type
    TGgml_abort_callback = function(
      const AData: pointer
    ): Boolean; cdecl;
  public type
    PInt32      = ^Int32;
    PPInt32     = ^PInt32;
    PSingle     = ^Single;
    PInt8       = ^ShortInt;

    llama_token = Int32;
    llama_pos   = Int32;
    llama_seq_id = Int32;

    PSingleArray = ^TSingleArray;
    TSingleArray = array[0..0] of Single;

    TTokenArray = array[0..0] of Int32;
    PTokenArray = ^TTokenArray;

    TPosArray = array[0..0] of Int32;
    PPosArray = ^TPosArray;

    TInt8Array = array[0..0] of Int8;
    PInt8Array = ^TInt8Array;

    TSeqId = Int32;

    TSeqIdArray = array[0..0] of TSeqId;
    PSeqIdArray = ^TSeqIdArray;

    TSeqIdPtrArray = array[0..0] of PSeqIdArray;
    PSeqIdPtrArray = ^TSeqIdPtrArray;

    llama_token_data = record
      id: llama_token;
      logit: Single;
      p: Single;
    end;

    TTokenDataArray = array[0..0] of llama_token_data;
    PTokenDataArray = ^TTokenDataArray;

    llama_token_data_array = record
      data: PTokenDataArray;
      size: Int32;
      sorted: Boolean;
    end;

    llama_model = Pointer;
    llama_context = Pointer;
    llama_vocab = Pointer;
    llama_memory_t = Pointer;

    llama_model_params = record
      devices: Pointer;
      tensor_buft_overrides: Pointer;
      n_gpu_layers: Int32;
      split_mode: TLlamaSplitMode;
      main_gpu: Int32;
      tensor_split: Pointer;
      progress_callback: Pointer;
      progress_callback_user_data: Pointer;
      llama_model_kv_override: Pointer;
      vocab_only: Boolean;
      use_mmap: Boolean;
      use_direct_io: Boolean;
      use_mlock: Boolean;
      check_tensors: Boolean;
      use_extra_bufts: Boolean;
      no_host: Boolean;
      no_alloc: Boolean;
    end;

    llama_context_params = record
      n_ctx: Int32;
      n_batch: Int32;
      n_ubatch: Int32;
      n_seq_max: Int32;
      n_threads: Int32;
      n_threads_batch: Int32;
      rope_scaling_type: TLlamaRopeScalingType;
      pooling_type: TLlamaPoolingType;
      llama_attention_type: TLlamaAttentionType;
      llama_flash_attn_type: TLlamaFlashAttentionType;
      rope_freq_base: Single;
      rope_freq_scale: Single;
      yarn_ext_factor: Single;
      yarn_attn_factor: Single;
      yarn_beta_fast: Single;
      yarn_beta_slow: Single;
      yarn_orig_ctx: Int32;
      defrag_thold: Single;
      cb_eval: Pointer;
      cb_eval_user_data: Pointer;
      type_k: Int32;
      type_v: Int32;
      abort_callback: TGgml_abort_callback;
      abort_callback_data: Pointer;
      embeddings: Boolean;
      offload_kqv: Boolean;
      no_perf: Boolean;
      op_offload: Boolean;
      swa_full: Boolean;
      kv_unified: Boolean;
      samplers: Pointer;
      n_samplers: NativeInt;
    end;

    llama_batch = record
      n_tokens: Int32;
      token: PTokenArray;
      embd: PSingle;
      pos: PPosArray;
      n_seq_id: PSeqIdArray;
      seq_id: PSeqIdPtrArray;
      logits: PInt8Array;
    end;

    llama_logit_array = PSingleArray;

    llama_chat_message = record
      role: PAnsiChar;
      content: PAnsiChar;
    end;
    Pllama_chat_message = ^llama_chat_message;
  private type
    TGgml_log_callback = procedure(
      level: TGgmlLogLevel;
      const text: PAnsiChar;
      user_data: Pointer
    ); cdecl;

    TGgml_backend_load_all = procedure; cdecl;

    TLlama_backend_init = procedure; cdecl;

    TLlama_max_devices = function: NativeUInt; cdecl;

    TLlama_model_load_from_file = function(
      const path: PAnsiChar;
      const params: llama_model_params
    ): llama_model; cdecl;

    TLlama_model_get_vocab = function(
      model: llama_model
    ): llama_vocab; cdecl;

    TLlama_init_from_model = function (
      model: llama_model;
      const params: llama_context_params
    ): llama_context; cdecl;

    TLlama_free = procedure(
      ctx: llama_context
    ); cdecl;

    TLlama_model_free = procedure(
      model: llama_model
    ); cdecl;

    TLlama_n_ctx = function(
      ctx: llama_context
    ): Int32; cdecl;

    TLlama_tokenize= function(
      vocab: llama_vocab;
      text: PAnsiChar;
      text_len: Int32;
      tokens: PInt32;
      n_max_tokens: Int32;
      add_special: Boolean;
      parse_special: Boolean
    ): Int32; cdecl;

    TLlama_batch_init = function(
      n_tokens: Int32;
      embd: Int32;
      n_seq_max: Int32
    ): llama_batch; cdecl;

    TLlama_batch_free = procedure(
      batch: llama_batch
    ); cdecl;

    TLlama_decode = function(
      ctx: llama_context;
      batch: llama_batch
    ): Int32; cdecl;

    TLlama_get_logits_ith = function(
      ctx: llama_context;
      i: Int32
    ): llama_logit_array; cdecl;

    TLlama_get_embeddings_ith = function(
      ctx: llama_context;
      i: Int32
    ): PSingle; cdecl;

    TLlama_get_memory = function(
      ctx: llama_context
    ): llama_memory_t; cdecl;

    TLlama_memory_seq_rm = function(
      mem: llama_memory_t;
      seq_id: llama_seq_id;
      p0: llama_pos;
      p1: llama_pos
    ): Boolean; cdecl;

    TLlama_vocab_n_tokens = function(
      vocab: llama_vocab
    ): Int32; cdecl;

    TLlama_model_n_embd = function(
      model: llama_model
    ): Int32; cdecl;

    TLlama_token_to_piece = function(
      vocab: llama_vocab;
      token: llama_token;
      buf: PAnsiChar;
      length: Int32;
      lstrip: Int32;
      special: Boolean
    ): Int32; cdecl;

    Tllama_log_set = procedure(
      ALogCallback: TGgml_log_callback;
      AUserData: Pointer
    ); cdecl;

    TLlama_model_meta_val_str = function(
      model: llama_model;
      key: PAnsiChar;
      buf: PAnsiChar;
      buf_size: Int32
    ): Int32; cdecl;

    TLlama_model_default_params = function: llama_model_params; cdecl;
    TLlama_context_default_params = function: llama_context_params; cdecl;

    TLlama_chat_apply_template = function(
      tmpl: PAnsiChar; // nil = usa template del modello
      chat: Pllama_chat_message;
      n_msg: Int32;
      add_assistant: Boolean;
      buf: PAnsiChar;
      buf_size: Int32
    ): Int32; cdecl;

    TLlama_encode = function(
      ctx: llama_context;
      batch: llama_batch
    ): Int32; cdecl;

    TLlama_vocab_eos = function(
      vocab: llama_vocab
    ): Int32; cdecl;

    TLlama_vocab_eot = function(
      vocab: llama_vocab
    ): Int32; cdecl;
  public
    ggml_backend_load_all: TGgml_backend_load_all;
    llama_backend_init: TLlama_backend_init;
    llama_max_devices: TLlama_max_devices;
    llama_model_load_from_file: TLlama_model_load_from_file;
    llama_model_get_vocab: TLlama_model_get_vocab;
    llama_init_from_model: TLlama_init_from_model;
    llama_free: TLlama_free;
    llama_model_free: TLlama_model_free;
    llama_n_ctx: TLlama_n_ctx;
    llama_tokenize: TLlama_tokenize;
    llama_batch_init: TLlama_batch_init;
    llama_batch_free: TLlama_batch_free;
    llama_decode: TLlama_decode;
    llama_get_logits_ith: TLlama_get_logits_ith;
    llama_get_embeddings_ith: TLlama_get_embeddings_ith;
    llama_get_memory: TLlama_get_memory;
    llama_memory_seq_rm: TLlama_memory_seq_rm;
    llama_vocab_n_tokens: TLlama_vocab_n_tokens;
    llama_model_n_embd: TLlama_model_n_embd;
    llama_token_to_piece: TLlama_token_to_piece;
    llama_log_set: Tllama_log_set;
    llama_model_meta_val_str: TLlama_model_meta_val_str;
    llama_model_default_params: TLlama_model_default_params;
    llama_context_default_params: TLlama_context_default_params;
    llama_chat_apply_template: TLlama_chat_apply_template;
    llama_encode: TLlama_encode;
    llama_vocab_eos: TLlama_vocab_eos;
    llama_token_eot: TLlama_vocab_eot;
  public
    procedure Load(const ALibPath: string);
    procedure Init;
    procedure Unload;
    procedure AddLogCallback(
      ALogCallback: TLlamaInstance.TGgml_log_callback;
      AUserData: Pointer
    );
  end;
{$ALIGN ON}

var
  LlamaInstance: TLlamaInstance;

implementation

const
{$IFDEF MSWINDOWS}
  LIB_LLAMA = 'llama.dll';
  LIB_GGML = 'ggml.dll';
{$ELSEIF DEFINED(OSX64)}
  LIB_LLAMA = 'libllama.dylib';
  LIB_GGML = 'libggml.dylib';
{$ELSE}
  LIB_LLAMA = 'libllama.so';
  LIB_GGML = 'libggml.so';
{$ENDIF}

{ TLlamaInstance }

function TLlamaInstance.GetProcAddress(
  const AHandle: THandle;
  const AProcName: string): Pointer;
begin
{$IFDEF MSWINDOWS}
  Result := Winapi.Windows.GetProcAddress(AHandle,
    PWideChar(WideString(AProcName)));
{$ELSE}
  Result := System.SysUtils.GetProcAddress(AHandle,
    PWideChar(WideString(AProcName)));
{$ENDIF}
end;

function TLlamaInstance.LoadLib(const LibName: string): THandle;
begin
  Result:=LoadDynLib(LibName, true);
end;

procedure TLlamaInstance.LoadLibProcs;
begin
  @ggml_backend_load_all:=GetProcAddress(FHandleGgmlLib,'ggml_backend_load_all');
  @llama_backend_init:=GetProcAddress(FHandleLLamaLib,'llama_backend_init');
  @llama_max_devices:=GetProcAddress(FHandleLLamaLib,'llama_max_devices');
  @llama_model_free:=GetProcAddress(FHandleLLamaLib,'llama_model_free');
  @llama_model_meta_val_str:=GetProcAddress(FHandleLLamaLib,'llama_model_meta_val_str');
  @llama_model_default_params:=GetProcAddress(FHandleLLamaLib,'llama_model_default_params');
  @llama_model_load_from_file:=GetProcAddress(FHandleLLamaLib,'llama_model_load_from_file');
  @llama_context_default_params:=GetProcAddress(FHandleLLamaLib,'llama_context_default_params');
  @llama_init_from_model:=GetProcAddress(FHandleLLamaLib,'llama_init_from_model');
  @llama_n_ctx:=GetProcAddress(FHandleLLamaLib,'llama_n_ctx');
  @llama_batch_init:=GetProcAddress(FHandleLLamaLib,'llama_batch_init');
  @llama_batch_free:=GetProcAddress(FHandleLLamaLib,'llama_batch_free');
  @llama_tokenize:=GetProcAddress(FHandleLLamaLib,'llama_tokenize');
  @llama_model_get_vocab:=GetProcAddress(FHandleLLamaLib,'llama_model_get_vocab');
  @llama_chat_apply_template:=GetProcAddress(FHandleLLamaLib,'llama_chat_apply_template');
  @llama_encode:=GetProcAddress(FHandleLLamaLib,'llama_encode');
  @llama_decode:=GetProcAddress(FHandleLLamaLib,'llama_decode');
  @llama_get_logits_ith:=GetProcAddress(FHandleLLamaLib,'llama_get_logits_ith');
  @llama_get_embeddings_ith:=GetProcAddress(FHandleLLamaLib,'llama_get_embeddings_ith');
  @llama_memory_seq_rm:=GetProcAddress(FHandleLLamaLib,'llama_memory_seq_rm');
  @llama_get_memory:=GetProcAddress(FHandleLLamaLib,'llama_get_memory');
  @llama_vocab_n_tokens:=GetProcAddress(FHandleLLamaLib,'llama_vocab_n_tokens');
  @llama_model_n_embd:=GetProcAddress(FHandleLLamaLib,'llama_model_n_embd');
  @llama_vocab_eos:=GetProcAddress(FHandleLLamaLib,'llama_vocab_eos');
  @llama_token_to_piece:=GetProcAddress(FHandleLLamaLib,'llama_token_to_piece');
  @llama_log_set:=GetProcAddress(FHandleLLamaLib,'llama_log_set');
  @llama_token_eot:=GetProcAddress(FHandleLLamaLib,'llama_token_eot');
end;

procedure TLlamaInstance.AddLogCallback(
  ALogCallback: TLlamaInstance.TGgml_log_callback;
  AUserData: Pointer);
begin
  llama_log_set(ALogCallback, AUserData);
end;

procedure TLlamaInstance.Init;
var
  PreviousDir: string;
begin
  PreviousDir:=GetCurrentDir;
  try
    SetCurrentDir(FLibPath);
    ggml_backend_load_all;
    llama_backend_init;
  finally
    SetCurrentDir(PreviousDir);
  end;
end;

procedure TLlamaInstance.Load(const ALibPath: string);
var
  PreviousDir: string;
begin
  if not TDirectory.Exists(ALibPath) then
    raise Exception.CreateFmt('Path "%s" not found.', [ALibPath]);
  PreviousDir:=GetCurrentDir;
  try
    SetCurrentDir(ALibPath);
    FLibPath:=ALibPath;
    FHandleGgmlLib:=LoadLib(LIB_GGML);
    FHandleLLamaLib:=LoadLib(LIB_LLAMA);
  finally
    SetCurrentDir(PreviousDir);
  end;
  LoadLibProcs;
end;

procedure TLlamaInstance.Unload;
begin
  FreeDynLib(FHandleLLamaLib);
  FreeDynLib(FHandleGgmlLib);
end;

initialization
  LlamaInstance:=Default(TLlamaInstance);
end.
