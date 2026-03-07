package shard

// DType represents tensor data types. Mirrors cowrie.DType for standalone use.
type DType uint8

const (
	DTypeUnknown  DType = 0x00
	DTypeFloat32  DType = 0x01
	DTypeFloat16  DType = 0x02
	DTypeFloat64  DType = 0x03
	DTypeInt8     DType = 0x04
	DTypeInt16    DType = 0x05
	DTypeInt32    DType = 0x06
	DTypeInt64    DType = 0x07
	DTypeUint8    DType = 0x08
	DTypeBFloat16 DType = 0x09
)

// DTypeName returns a human-readable name for a DType.
func DTypeName(dt DType) string {
	switch dt {
	case DTypeFloat32:
		return "float32"
	case DTypeFloat16:
		return "float16"
	case DTypeFloat64:
		return "float64"
	case DTypeInt8:
		return "int8"
	case DTypeInt16:
		return "int16"
	case DTypeInt32:
		return "int32"
	case DTypeInt64:
		return "int64"
	case DTypeUint8:
		return "uint8"
	case DTypeBFloat16:
		return "bfloat16"
	default:
		return "unknown"
	}
}
