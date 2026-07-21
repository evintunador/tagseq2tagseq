import 'package:flutter/material.dart';  // pub dep -> external, NO edge
import 'util/helper.dart';                // -> lib/util/helper.dart
import 'models/user.dart' show User;      // combinator stripped -> lib/models/user.dart
import './consts.dart' as c;              // explicit ./ + alias -> lib/consts.dart

export 'models/user.dart';                // re-export -> lib/models/user.dart

void main() {
  final u = User();
  print(helper() + u.id + c.value);
  runApp();
}
