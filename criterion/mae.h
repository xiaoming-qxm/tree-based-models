// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef MAE_H_
#define MAE_H_

#include "regression_criterion.h"

namespace tree_based_model {

// Mean absolute error impurity criterion
// MAE = (1 / n)*(\sum_i |y_i - f_i|), where y_i
// is the true value and f_i is the predicted value
class MAE : public RegressionCriterion {
public:
  MAE();

};

}

#endif // MAE_H_
