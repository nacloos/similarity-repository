package rdm

#rdms_from_dict: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Rdm Dict
	rdm_dict: _

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.rdms_from_dict"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
