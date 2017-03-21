// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef MODEL_FACTORY_H_
#define MODEL_FACTORY_H_

#include "../tree/id3.h"
#include <string>
#include <iostream>

namespace tree_based_model {

class ModelFactory {
public:
  TreeModel* CreateModel(const std::string type, const int num_classes, const int num_feature) {
    if(type.compare("id3") == 0)
      return new ID3(num_classes, num_feature);
    else if(type.compare("cart") == 0)
      exit(1);
    else
      exit(1);
  }

};

}


#endif // MODEL_FACTORY_H_
