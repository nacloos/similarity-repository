package rdm

#load_rdm: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Filename
	filename: _

	// File Type
	file_type?: _ | *null

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.load_rdm"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
