// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef CLASSIFICATION_TREE_H_
#define CLASSIFICATION_TREE_H_

#include "tree.h"

namespace tree_based_model {

class ClassificationTree : public Tree {
public:
  ClassificationTree() = default;
  virtual ~ClassificationTree() = default;
};

}

#endif // CLASSIFICATION_TREE_H_
