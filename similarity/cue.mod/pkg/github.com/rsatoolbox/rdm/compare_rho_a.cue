package rdm

#compare_rho_a: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Rdm1
	rdm1: _

	// Rdm2
	rdm2: _

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.compare_rho_a"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
