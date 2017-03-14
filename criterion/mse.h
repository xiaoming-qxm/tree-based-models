// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef MSE_H_
#define MSE_H_

#include "regression_criterion.h"

using namespace tree_based_model {

// Mean squared error impurity criterion
class MSE : RegressionCriterion {
public:
  MSE();

};

}

#endif // MSE_H_
