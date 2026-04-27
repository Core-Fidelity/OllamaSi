package discover

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework CoreGraphics -framework Metal
#include "gpu_info_darwin.h"
*/
import "C"

import (
	"log/slog"
	"syscall"

	"github.com/ollama/ollama/format"
)

const (
	metalMinimumMemory = 512 * format.MebiByte
)

func GetCPUMem() (memInfo, error) {
	// On Apple Silicon, Metal uses unified memory. recommendedMaxWorkingSetSize
	// is Metal's guidance on how much it should use. The Go scheduler should
	// treat this as the effective GPU memory budget, not total physical RAM,
	// to avoid overcommitting what Metal can actually allocate.
	total := uint64(C.getPhysicalMemory())
	recommendedVRAM := uint64(C.getRecommendedMaxVRAM())
	if recommendedVRAM > 0 && recommendedVRAM < total {
		total = recommendedVRAM
	}

	free := uint64(C.getFreeMemory())
	allocatedVRAM := uint64(C.getCurrentAllocatedVRAM())
	if allocatedVRAM > 0 && allocatedVRAM < free {
		free -= allocatedVRAM
	}

	return memInfo{
		TotalMemory: total,
		FreeMemory:  free,
		// FreeSwap omitted as Darwin uses dynamic paging
	}, nil
}

func GetCPUDetails() []CPU {
	query := "hw.perflevel0.physicalcpu"
	perfCores, err := syscall.SysctlUint32(query)
	if err != nil {
		slog.Warn("failed to discover physical CPU details", "query", query, "error", err)
	}
	query = "hw.perflevel1.physicalcpu"
	efficiencyCores, _ := syscall.SysctlUint32(query) // On x86 xeon this wont return data

	// Determine thread count
	query = "hw.logicalcpu"
	logicalCores, _ := syscall.SysctlUint32(query)

	return []CPU{
		{
			CoreCount:           int(perfCores + efficiencyCores),
			EfficiencyCoreCount: int(efficiencyCores),
			ThreadCount:         int(logicalCores),
		},
	}
}

func IsNUMA() bool {
	// numa support in ggml is linux only
	return false
}
