package rdm

#positive_transform: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Rdms
	rdms: _

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.positive_transform"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
