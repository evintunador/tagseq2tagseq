package store

import "example.com/proj/util"

type Store struct{ name string }

func New() *Store { return &Store{name: util.Default()} }
func (s *Store) Name() string { return s.name }
