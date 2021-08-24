from nasws.cnn.search_space.darts.darts_search_space import Genotype, DartsModelSpec
from nasws.cnn.policy.nao_policy.utils_for_darts import NAOParsingDarts

def print_arch_to_evaluate(arch_pool, space, parser: NAOParsingDarts, top_k=3):
    eval_archs = []

    for k in arch_pool.keys():
        if isinstance(k, str):
            if k.startswith('Genotype'):
                model_spec = DartsModelSpec.from_darts_genotype(eval(k))
            else:
                model_spec = parser.parse_arch_to_model_spec(eval(k))
            
        elif isinstance(k, Genotype):
            model_spec = DartsModelSpec.from_darts_genotype(k)
        elif isinstance(k, DartsModelSpec):
            model_spec = k
        else:
            raise NotImplementedError(f'arch pool key type not supported: {k}')
    
        _id = space.topology_to_id(model_spec)
        if _id is None:
            print(f'new model with spec {model_spec}')
            eval_archs.append(parser.parse_model_spec_to_arch(model_spec))
        if len(eval_archs) >= top_k:
            break
        # else:
            # print('old model ...')
    print('\'\n\''.join(map(str, eval_archs)))
