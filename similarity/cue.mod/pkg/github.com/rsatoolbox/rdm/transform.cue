package rdm

#transform: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Rdms
	rdms: _

	// Fun
	fun: _

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.transform"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
