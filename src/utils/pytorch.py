from torch.nn import Module

from utils.string import to_human_readable


def get_total_params(model: Module, print_table: bool = False, sort_desc: bool = False) -> int or None:
    """
    Get total number of parameters from given nn.Module.
    :param model: model to count parameters for
    :param print_table: if True prints counts for every sub-module and returns None, else returns total count only
    :param sort_desc: if True sorts array in DESC order wrt to parameter count before printing
    :return: total number of parameters if $print$ is set to False, else prints counts and returns nothing
    """
    total_count_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = 0
    count_dict = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        count_dict.append({
            'name': name,
            'count': count,
            'count_hr': to_human_readable(count),
            'prc': '%.2f %%' % (count / total_count_orig)
        })
        total_count += count

    assert total_count_orig == total_count, "Should be equal..."

    if print_table is True:
        from json import dumps
        from prettytable import from_json

        if sort_desc:
            count_dict = sorted(count_dict, key=lambda k: k['count'], reverse=True)

        table = from_json(dumps(count_dict, indent=4))
        table.field_names = ["Module", "Count", "Count Human", "Percentage"]

        print(table.get_string(fields=["Module", "Count Human", "Percentage"]))
        print(f"Total Trainable Params: {to_human_readable(total_count)}")
        return None

    return total_count
