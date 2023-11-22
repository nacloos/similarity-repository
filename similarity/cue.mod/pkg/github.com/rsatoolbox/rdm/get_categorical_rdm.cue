package rdm

#get_categorical_rdm: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Category Vector
	category_vector: _

	// Category Name
	category_name?: _ | *"category"

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.get_categorical_rdm"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
