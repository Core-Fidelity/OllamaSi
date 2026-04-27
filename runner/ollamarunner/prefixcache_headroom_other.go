//go:build !darwin

package ollamarunner

func availablePrefixCacheHeadroomBytes() (uint64, bool) {
	return 0, false
}
