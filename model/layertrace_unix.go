//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package model

import (
	"log/slog"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/model/input"
	"golang.org/x/sys/unix"
)

var traceLayerPageFaults = envconfig.Bool("OLLAMA_TRACE_PAGE_FAULTS")()

type pageFaultSample struct {
	minflt int64
	majflt int64
}

func samplePageFaults() (pageFaultSample, bool) {
	var ru unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_SELF, &ru); err != nil {
		return pageFaultSample{}, false
	}

	return pageFaultSample{
		minflt: ru.Minflt,
		majflt: ru.Majflt,
	}, true
}

// TraceLayer measures page faults and elapsed time around a single layer forward call.
func TraceLayer(modelName string, layer int, batch input.Batch, fn func()) {
	if !traceLayerPageFaults {
		fn()
		return
	}

	before, ok := samplePageFaults()
	started := time.Now()
	fn()
	after, ok2 := samplePageFaults()
	if !ok || !ok2 {
		return
	}

	posFirst, posLast := int32(0), int32(0)
	if len(batch.Positions) > 0 {
		posFirst = batch.Positions[0]
		posLast = batch.Positions[len(batch.Positions)-1]
	}

	slog.Info("layer page faults",
		"model", modelName,
		"layer", layer,
		"tokens", len(batch.Positions),
		"positions", len(batch.Positions),
		"pos_first", posFirst,
		"pos_last", posLast,
		"minflt", after.minflt-before.minflt,
		"majflt", after.majflt-before.majflt,
		"elapsed_ms", float64(time.Since(started))/float64(time.Millisecond),
	)
}
