package Set

type Set map[int]bool

func (s Set) Add(id int) {
	s[id] = true
}

func NewSet() Set {
	return make(map[int]bool)
}

func (s Set) Copy() Set {
	tmp := NewSet()
	for v := range s {
		tmp.Add(v)
	}
	return tmp
}

func (s Set) Size() int {
	return len(s)
}

func (s Set) Merge(an Set) {
	for v := range an {
		s[v] = true
	}
}

func (s Set) Has(v int) bool {
	_, ok := s[v]
	return ok
}

func (s Set) Remove(v int) {
	delete(s, v)
}

func (s Set) HasIntersection(an Set) bool {
	var a, b Set
	if s.Size() < an.Size() {
		a = s
		b = an
	} else {
		a = an
		b = s
	}

	for v := range a {
		if b.Has(v) {
			return true
		}
	}
	return false
}

func (s Set) Clear() {
	for v := range s {
		s.Remove(v)
	}
}

func (s Set) Top() int {
	for v := range s {
		return v
	}
	return -1
}
