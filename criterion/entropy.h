// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef ENTROPY_H_
#define ENTROPY_H_

#include "classification_criterion.h"

using namespace tree_based_model {

// Cross Entropy impurity criterion
class Entropy : ClassificationCriterion {
public:
  Entropy();

};

}

#endif // ENTROPY_H_
