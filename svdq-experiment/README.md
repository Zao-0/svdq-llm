Public APIs and usage

data_gen.py:
- get_cali_data(tok, seed: int = 0, cn: int = 100, eu: int = 100, device: str = "cpu") -> Dict[str, List[torch.Tensor]]
- get_veri_data(tok, seed: int = 0, cn: int = 1000, cn_len: int = 1024, eu: int = 1600, eu_len: int = 512, device: str = "cpu") -> Dict[str, List[torch.Tensor]]

smooth_util.py:
- SmoothCfg: 
    seed: int = 0
    cn: int = 100
    eu: int = 160

    device: str = "cuda:0"                 # where model runs if device_map is None
    device_for_inputs: Optional[str] = None  # where input_ids are placed; default inferred

    alpha: float = 0.5
    eps: float = 1e-6
    act_stat: str = "absmax"               # "absmax" or "meanabs"

    torch_dtype: Any = "auto"
    device_map: Optional[Any] = None       # if not None, accelerate dispatch

    safe_serialization: bool = True
    max_shard_size: str = "10GB"
- smoothen_model(model_dir: str, smt_cfg: Dict[str, Any], save_path: str, label: str)
- load_smoothed_model(model_dir: str)

ppl_evaluator.py
- evl(model: nn.Module, tokenizer: Any, result_path: str, scenario_name: str = None, limit: Optional[int | float] = None, batch_size: int | str = "auto") -> Dict[str, float]

fake_svdq.py
- fake_op(model: nn.Module, svdq_config: Dict[str, Any]) -> nn.Module

