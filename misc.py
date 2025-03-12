import re
import contextlib
import numpy as np
import torch
import warnings
import dnnlib
import functools
import persistence


# 初始化一个缓存字典，用于存储常量的张量
_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    """
    创建一个常量张量，并使用缓存机制避免重复创建相同的张量。

    参数:
        value (数值或可转换为数组的对象): 要转换为张量的值。
        shape (tuple 或 list, 可选): 目标张量的形状。如果提供，将广播原始值以匹配此形状。
        dtype (torch.dtype, 可选): 张量的数据类型。如果未提供，将使用默认数据类型。
        device (torch.device, 可选): 张量所在的设备。如果未提供，将使用 CPU。
        memory_format (torch.memory_format, 可选): 张量的内存格式。如果未提供，将使用默认的连续内存格式。

    返回:
        torch.Tensor: 创建的常量张量。
    """
    # 将输入值转换为 NumPy 数组，以确保一致性和可处理性
    value = np.asarray(value)

    # 如果提供了形状参数，则将其转换为元组类型
    if shape is not None:
        shape = tuple(shape)

    # 如果未提供数据类型，则使用 PyTorch 的默认数据类型
    if dtype is None:
        dtype = torch.get_default_dtype()

    # 如果未提供设备，则默认使用 CPU
    if device is None:
        device = torch.device("cpu")

    # 如果未提供内存格式，则默认使用连续内存格式
    if memory_format is None:
        memory_format = torch.contiguous_format

    # 构建一个键，用于缓存中查找或存储该张量
    key = (
        value.shape,      # NumPy 数组的形状
        value.dtype,      # NumPy 数组的数据类型
        value.tobytes(),  # NumPy 数组的字节表示，用于唯一标识内容
        shape,            # 目标形状
        dtype,            # 目标数据类型
        device,           # 目标设备
        memory_format     # 目标内存格式
    )

    # 从缓存中尝试获取已存在的张量
    tensor = _constant_cache.get(key, None)
    
    if tensor is None:
        # 如果缓存中不存在，则创建新的张量
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        # 如果提供了形状参数，则广播张量以匹配目标形状
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        # 设置张量的内存格式
        tensor = tensor.contiguous(memory_format=memory_format)
        # 将新创建的张量存储到缓存中
        _constant_cache[key] = tensor

    # 返回缓存中的张量或新创建的张量
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    """
    创建一个与参考张量具有相同数据类型和设备的常量张量。

    参数:
        ref (torch.Tensor): 参考张量，用于继承数据类型和设备。
        value (数值或可转换为数组的对象): 要转换为张量的值。
        shape (tuple 或 list, 可选): 目标张量的形状。如果提供，将广播原始值以匹配此形状。
        dtype (torch.dtype, 可选): 张量的数据类型。如果未提供，将继承自参考张量。
        device (torch.device, 可选): 张量所在的设备。如果未提供，将继承自参考张量。
        memory_format (torch.memory_format, 可选): 张量的内存格式。如果未提供，将使用默认的连续内存格式。

    返回:
        torch.Tensor: 创建的常量张量。
    """
    # 如果未提供数据类型，则继承自参考张量的数据类型
    if dtype is None:
        dtype = ref.dtype
    # 如果未提供设备，则继承自参考张量的设备
    if device is None:
        device = ref.device
    # 调用 constant 函数创建张量，并传递继承的参数
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)


@functools.lru_cache(None)
def pinned_buf(shape, dtype):
    """
    在锁页内存中创建临时张量，并使用缓存机制避免重复创建相同的张量。

    参数:
        shape (tuple 或 list): 张量的形状。
        dtype (torch.dtype): 张量的数据类型。

    返回:
        torch.Tensor: 在锁页内存中创建的临时张量。
    """
    # 使用 torch.empty 创建空张量，并将其锁定到内存中（pin_memory）
    return torch.empty(shape, dtype=dtype).pin_memory()


# 尝试导入 torch._assert 方法，如果失败则使用 torch.Assert 方法
try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0


# 尝试导入 torch.nan_to_num 函数，如果失败则定义一个自定义的 nan_to_num 函数
try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:
    # 对于较旧的版本，手动实现 nan_to_num 功能
    def nan_to_num(
        input, nan=0.0, posinf=None, neginf=None, *, out=None
    ):  
        """
        将输入张量中的 NaN、+inf 和 -inf 替换为指定的值。

        参数:
            input (torch.Tensor): 输入张量。
            nan (float, 可选): 用于替换 NaN 的值，默认为 0.0。
            posinf (float, 可选): 用于替换正无穷大的值。如果未提供，将使用张量数据类型的最大有限值。
            neginf (float, 可选): 用于替换负无穷大的值。如果未提供，将使用张量数据类型的最小有限值。
            out (torch.Tensor, 可选): 输出张量，用于存储结果。

        返回:
            torch.Tensor: 替换后的张量。
        """
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            # 获取数据类型允许的最大值
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            # 获取数据类型允许的最小值
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        # 使用 unsqueeze 扩展维度，使用 nansum 替换 NaN 为 0，然后使用 clamp 将值限制在 [neginf, posinf] 之间
        return torch.clamp(
            input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out
        )


# 再次尝试导入 torch._assert 方法，如果失败则使用 torch.Assert 方法
try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0


@contextlib.contextmanager
def suppress_tracer_warnings():
    """
    上下文管理器，用于在 torch.jit.trace() 中临时抑制特定的警告。

    使用方法:
        with suppress_tracer_warnings():
            # 执行可能产生警告的代码
            ...
    """
    # 定义一个过滤规则，忽略 torch.jit.TracerWarning 类型的警告
    flt = ("ignore", None, torch.jit.TracerWarning, None, 0)
    # 将过滤规则插入到警告过滤器列表的开头
    warnings.filters.insert(0, flt)
    yield
    # 移除之前插入的过滤规则，恢复原始状态
    warnings.filters.remove(flt)


def assert_shape(tensor, ref_shape):
    """
    确保张量的形状与给定的参考形状匹配。

    参数:
        tensor (torch.Tensor): 要检查形状的张量。
        ref_shape (list 或 tuple): 参考形状，元素为整数或 None。None 表示该维度的大小可以变化。

    异常:
        AssertionError: 如果张量的形状与参考形状不匹配。
    """
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(
                f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
            )


def profiled_function(fn):
    """
    函数装饰器，用于记录函数的执行时间。

    参数:
        fn (callable): 需要记录的函数。

    返回:
        callable: 装饰后的函数。
    """
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)

    decorator.__name__ = fn.__name__
    return decorator


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        """
        初始化 InfiniteSampler。

        参数:
            dataset (torch.utils.data.Dataset): 要采样的数据集。
            rank (int, 可选): 当前进程的排名，默认为 0。
            num_replicas (int, 可选): 进程总数，默认为 1。
            shuffle (bool, 可选): 是否打乱数据集顺序，默认为 True。
            seed (int, 可选): 随机种子，默认为 0。
            window_size (float, 可选): 窗口大小，范围 [0, 1]，用于决定是否交换样本，默认为 0.5。
        """
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset      # 数据集
        self.rank = rank            # 当前进程排名
        self.num_replicas = num_replicas  # 进程总数
        self.shuffle = shuffle      # 是否打乱顺序
        self.seed = seed            # 随机种子
        self.window_size = window_size  # 窗口大小

    def __iter__(self):
        """
        生成无限长的样本索引迭代器。

        返回:
            iterator: 无限长的样本索引迭代器。
        """
        # 生成数据集的索引数组
        order = np.arange(len(self.dataset))
        # 随机数生成器初始化为 None
        rnd = None
        # 窗口大小初始化为 0
        window = 0
        if self.shuffle:
            # 使用种子初始化随机数生成器
            rnd = np.random.RandomState(self.seed)
            # 打乱索引顺序
            rnd.shuffle(order)
            # 计算窗口大小，向上取整
            window = int(np.rint(order.size * self.window_size))

        # 初始化索引
        idx = 0
        while True:
            # 计算当前索引在打乱后的顺序中的位置
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                # 如果当前索引由该进程处理，则生成该索引
                yield order[i]
            if window >= 2:
                # 计算要交换的另一个索引
                j = (i - rnd.randint(window)) % order.size
                # 交换两个索引对应的样本
                order[i], order[j] = order[j], order[i]
            # 索引递增
            idx += 1


def params_and_buffers(module):
    """
    获取模块的所有参数和缓冲区。

    参数:
        module (torch.nn.Module): 要获取参数和缓冲区的模块。

    返回:
        list: 包含所有参数和缓冲区的列表。
    """
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    """
    获取模块的所有命名参数和命名缓冲区。

    参数:
        module (torch.nn.Module): 要获取命名参数和缓冲区的模块。

    返回:
        list: 包含所有命名参数和命名缓冲区的列表。
    """
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    """
    将源模块的参数和缓冲区复制到目标模块。

    参数:
        src_module (torch.nn.Module): 源模块，包含要复制的参数和缓冲区。
        dst_module (torch.nn.Module): 目标模块，将被复制参数和缓冲区。
        require_all (bool, 可选): 如果为 True，则要求目标模块包含源模块的所有参数和缓冲区，默认为 False。
    """
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    # 获取源模块的命名参数和缓冲区，并转换为字典
    src_tensors = dict(named_params_and_buffers(src_module))
    # 遍历目标模块的命名参数和缓冲区
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors: 
            # 将源模块的参数或缓冲区复制到目标模块
            tensor.copy_(src_tensors[name]) 


@contextlib.contextmanager
def ddp_sync(module, sync):
    """
    上下文管理器，用于控制 DistributedDataParallel 的同步行为。

    参数:
        module (torch.nn.Module): 需要控制同步行为的模块。
        sync (bool): 如果为 True，则启用同步；否则，禁用同步。

    使用方法:
        with ddp_sync(module, sync=True):
            # 执行需要同步的代码
            ...
    """
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        # 如果启用同步或模块不是 DDP，则直接执行
        yield
    else:
        # 否则，使用 no_sync 上下文管理器禁用同步
        with module.no_sync():
            yield


def check_ddp_consistency(module, ignore_regex=None):
    """
    检查 DistributedDataParallel 模块在不同进程之间的一致性。

    参数:
        module (torch.nn.Module): 需要检查的模块。
        ignore_regex (str, 可选): 正则表达式字符串，用于指定需要忽略的参数或缓冲区的名称，默认为 None。

    异常:
        AssertionError: 如果模块在不同进程之间不一致。
    """
    assert isinstance(module, torch.nn.Module)
    # 遍历模块的命名参数和缓冲区
    for name, tensor in named_params_and_buffers(module):
        # 构建完整的参数或缓冲区名称
        fullname = type(module).__name__ + "." + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            # 如果名称匹配忽略的正则表达式，则跳过
            continue
        # 分离张量，避免梯度计算
        tensor = tensor.detach()
        # 如果是浮点张量，则替换 NaN 和 Inf
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        # 克隆张量以进行广播
        other = tensor.clone()
        # 将张量从 rank 0 广播到其他进程
        torch.distributed.broadcast(tensor=other, src=0) 
        # 确保所有进程的张量一致
        assert (tensor == other).all(), fullname


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    """
    打印模块的摘要信息，包括参数数量、缓冲区数量、输出形状和数据类型。

    参数:
        module (torch.nn.Module): 需要打印摘要信息的 PyTorch 模块。
        inputs (tuple 或 list): 模块的输入，可以是元组或列表。
        max_nesting (int, 可选): 最大嵌套深度，用于控制打印的详细程度，默认为 3。
        skip_redundant (bool, 可选): 是否跳过冗余的条目，默认为 True。

    返回:
        tuple: 模块的输出。
    """
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # 初始化用于存储钩子结果的列表和嵌套深度计数器
    entries = []
    nesting = [0]

    # 定义前向传播前的钩子函数
    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    # 定义前向传播后的钩子函数
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))

    # 为模块的所有子模块注册前向传播前的钩子
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    # 为模块的所有子模块注册前向传播后的钩子
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # 运行模块的前向传播
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # 识别唯一的输出、参数和缓冲区
    tensors_seen = set() # 用于存储已见过的张量 ID
    for e in entries:
        # 获取唯一的参数，排除重复的参数
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        # 获取唯一的缓冲区，排除重复的缓冲区
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        # 获取唯一的输出，排除重复的输出
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        # 更新已见过的张量 ID 集合
        tensors_seen |= {
            id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs
        }

    # 如果需要跳过冗余的条目，则过滤掉没有唯一参数、缓冲区或输出的条目
    if skip_redundant:
        entries = [
            e
            for e in entries
            if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)
        ]

    # 构建表格的标题行
    rows = [
        [type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]
    ]

    # 添加分隔行
    rows += [["---"] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
        rows += [
            [
                name + (":0" if len(e.outputs) >= 2 else ""),
                str(param_size) if param_size else "-",
                str(buffer_size) if buffer_size else "-",
                (output_shapes + ["-"])[0],
                (output_dtypes + ["-"])[0],
            ]
        ]
        for idx in range(1, len(e.outputs)):
            rows += [
                [name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]
            ]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print(
            "  ".join(
                cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
            )
        )
    print()
    return outputs


import abc
@persistence.persistent_class
class ActivationHook(abc.ABC):
    """
    抽象基类，用于定义激活钩子。

    子类需要实现 __call__ 方法，以定义钩子的行为。
    """
    def __init__(self,  modules_to_watch):
        """
        初始化激活钩子。

        参数:
            modules_to_watch (list 或 str): 需要监视的模块名称列表，或特殊字符串 'all' 表示监视所有模块。
        """
        self.modules_to_watch = modules_to_watch

        self._hook_result = {}

    @property
    def hook_result(self):
        """
        获取钩子的结果。

        返回:
            dict: 包含钩子结果的字典。
        """
        return self._hook_result

    @abc.abstractmethod
    def __call__(self, module, input, output):
        """
        定义钩子的行为。

        参数:
            module (torch.nn.Module): 当前模块。
            input (tuple): 模块的输入。
            output (torch.Tensor 或 tuple): 模块的输出。
        """
        pass
            
    def watch(self, models_dict):
        """
        注册钩子到指定的模块。

        参数:
            models_dict (dict): 包含模型名称和模型实例的字典。
        """
        acc = []
        # 遍历所有模型和模块
        for k in models_dict:
            model = models_dict[k]
            for name, module in model.named_modules():
                if self.modules_to_watch == 'all':
                    module._hook_name = name
                else:
                    for mw in self.modules_to_watch:
                        if mw in name and name not in acc:
                            module._hook_name = k + '.' + name
                            acc.append(name)


    def clear(self):
        """
        清空钩子的结果。
        """
        self._hook_result = {}


@persistence.persistent_class
class ActivationMagnitudeHook(ActivationHook):
    """
    子类，用于计算激活的幅度。

    继承自 ActivationHook，并实现了 __call__ 方法。
    """
    def __init__(self, modules_to_watch='all'):
        """
        初始化激活幅度钩子。

        参数:
            modules_to_watch (list 或 str, 可选): 需要监视的模块名称列表，或特殊字符串 'all'，默认为 'all'。
        """
        super().__init__(modules_to_watch)

    def __call__(self, module, input, output):
        """
        计算激活的幅度，并存储结果。

        参数:
            module (torch.nn.Module): 当前模块。
            input (tuple): 模块的输入。
            output (torch.Tensor): 模块的输出。
        """
        if hasattr(module, '_hook_name'):
            # 仅跟踪已注册的模块
            if isinstance(output, torch.Tensor):
                output_ = output.detach()
                # 计算激活的幅度，并防止溢出
                self._hook_result['activations/' + module._hook_name + '_magnitude_div_10000'] = (output_/10000).flatten(1).norm(1).mean().item() #prevent overflow
            else:
                self._hook_result['activations/' + module._hook_name + '_magnitude_div_10000'] = 0

