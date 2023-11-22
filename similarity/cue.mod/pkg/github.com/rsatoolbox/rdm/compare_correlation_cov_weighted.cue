package rdm

#compare_correlation_cov_weighted: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Rdm1
	rdm1: _

	// Rdm2
	rdm2: _

	// Sigma K
	sigma_k?: _ | *null

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.compare_correlation_cov_weighted"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
