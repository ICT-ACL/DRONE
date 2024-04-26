#include "utils.h"

long long master2mirror_forward_total[2], mirror2master_forward_total[2], master2mirror, mirror2master,
        master2mirror_backward_total[2], mirror2master_backward_total[2];

int current_epoch;
float mean_accuracy;
bool update_threshold;