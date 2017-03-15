// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef REGRESSION_CRITERION_H_
#define REGRESSION_CRITERION_H_

#include "criterion.h"

namespace tree_based_model {

class RegressionCriterion : public Criterion() {
public:
  RegressionCriterion() = default;
  virtual ~RegressionCriterion() = default;

};

}

#endif // REGRESSION_CRITERION_H_
