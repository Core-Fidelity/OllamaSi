//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris

package model

import "github.com/ollama/ollama/model/input"

// TraceLayer is a no-op on platforms that do not expose rusage counters.
func TraceLayer(_ string, _ int, _ input.Batch, fn func()) {
	fn()
}
