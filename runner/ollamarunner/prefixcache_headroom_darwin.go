//go:build darwin

package ollamarunner

/*
#include <mach/mach.h>
#include <mach/host_info.h>

// Sum free + speculative + inactive pages to estimate available RAM.
// macOS aggressively pages idle memory to inactive; free_count alone
// is typically near zero even on idle systems. Counting only free pages
// would falsely flag the system as memory-constrained and disable
// prefix-cache hydration on every load.
static uint64_t ollama_prefix_cache_available_memory_bytes(void) {
	mach_port_t host_port = mach_host_self();
	vm_size_t pagesize = 0;
	vm_statistics64_data_t vm_stat;
	mach_msg_type_number_t host_size = HOST_VM_INFO64_COUNT;

	if (host_page_size(host_port, &pagesize) != KERN_SUCCESS) {
		return 0;
	}
	if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
		return 0;
	}

	return ((uint64_t)vm_stat.free_count +
		(uint64_t)vm_stat.speculative_count +
		(uint64_t)vm_stat.inactive_count) * (uint64_t)pagesize;
}
*/
import "C"

func availablePrefixCacheHeadroomBytes() (uint64, bool) {
	return uint64(C.ollama_prefix_cache_available_memory_bytes()), true
}
