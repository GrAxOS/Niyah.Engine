# KHZ Graph Repair Report

## Summary
duplicate_node_ids: 90
dangling_edges: 6
missing_pedagogy: 501

## Dangling edges

id source                               target                        missing_source missing_target file
-- ------                               ------                        -------------- -------------- ----
   n_mlkem_768_pqc_key_encapsulation    n_fips_203_ml_kem_standard             False           True chunks\khawrizm_gra…
   n_mlkem_1024_pqc_key_encapsulation   n_fips_203_ml_kem_standard             False           True chunks\khawrizm_gra…
   n_xmss_rfc_8391_hash_tree_signatures n_ietf_rfc_8391_xmss_standard          False           True chunks\khawrizm_gra…
   n_mlkem_768_pqc_kem_algorithm        n_fips_203_ml_kem_standard             False           True chunks\khawrizm_gra…
   n_mlkem_1024_pqc_kem_algorithm       n_fips_203_ml_kem_standard             False           True chunks\khawrizm_gra…
   n_mlkem_512_pqc_kem_algorithm        n_fips_203_ml_kem_standard             False           True chunks\khawrizm_gra…


## Duplicate node IDs
n_acid_wal_persistence
n_amd_sev_snp_confidential_computing_enclave
n_arabert_contextual_language_model
n_arabert_pretraining_transformer
n_arabic_named_entity_linking_marbert
n_arabic_named_entity_recognition_camel_tools
n_arabic_stemming_khoja
n_bilingual_arabic_entity_aligner
n_bliss_graph_canonicalization_algorithm
n_bloom_filter_edge_pruning
n_camel_tools_morphological_analyzer
n_camel_tools_morphological_disambiguator
n_camel_tools_morphological_generator
n_camel_tools_named_entity_recognizer
n_cross_lingual_entity_alignment
n_cuda_graph_execution_api
n_cxl_3_1_dynamic_capacity_device_dcd
n_cxl_3_1_mld_multi_logical_device
n_differential_privacy_graph_masking
n_differential_privacy_privacy_budget_tracker
n_farasa_arabic_morphological_segmenter
n_farasa_lemmatizer_arabic
n_fips_203_ml_kem_key_encapsulation_standard
n_fpga_graph_traversal_accelerator
n_graph_blas_linear_algebra_api
n_graph_embedding_node2vec
n_graph_partitioning_fennel
n_graph_rag_retrieval
n_graph_sage_inductive_sampling
n_graph_schema_evolution
n_graphblas_semiring_algebraic_structure
n_graphblas_semiring_min_plus_tropical
n_graphblas_sparse_matrix_vector_mult_gemv
n_hnsw_hierarchical_navigable_small_world
n_hnsw_pq_compression
n_intel_tdx_trust_domain_extensions
n_iso_iec_27001_sovereign_compliance
n_iso_iec_27701_privacy_information_management
n_iso_iec_39788_gql_standard
n_iso_iec_39788_graph_query
n_katz_centrality_node_ranking
n_khoja_stemming_algorithm_arabic
n_kyber_crystals_post_quantum_key_encapsulation
n_labse_language_agnostic_bert_embeddings
n_louvain_community_detection_algorithm
n_madamira_arabic_morphological_analyzer
n_merkle_dag_audit_provenance_chain
n_merkle_dag_provenance_tree
n_mldsa_44_pqc_signature_algorithm
n_mldsa_65_pqc_signature_algorithm
n_mldsa_87_pqc_signature_algorithm
n_mpt_merkle_patricia_trie_state
n_non_volatile_memory_express_nvme_direct
n_nvlink_4_direct_gpu_interconnect
n_nvme_of_rdma_fabric
n_openmp_simd_parallel_processing
n_post_quantum_bike_kem_protocol
n_post_quantum_falcon_signature_scheme
n_qutrub_arabic_verb_conjugator
n_rabin_karp_rolling_hash_deduplication
n_rdf_owl_ontology_mapper
n_rdf_star_triple_annotation_extension
n_rdma_rocev2_transport_protocol
n_roaring_bitmap_indexing_engine
n_shacl_compact_syntax_shaclc
n_shacl_sparql_constraint_component
n_simd_avx_512_vector_extensions
n_simd_vectorized_similarity_kernel
n_slh_dsa_sha2_128_fast_algorithm
n_slh_dsa_sha2_256_fast_algorithm
n_slh_dsa_shake_128_fast_algorithm
n_slh_dsa_shake_256_fast_algorithm
n_spdm_1_3_security_protocol
n_temporal_graph_neural_network_tgn
n_temporal_graph_snapshot
n_tensor_processing_unit_tpu_v5p
n_vf2_plus_subgraph_isomorphism_kernel
n_vf2_subgraph_isomorphism
n_w3c_n_quads_rdf_dataset_format
n_w3c_owl2_ql_profile_standard
n_w3c_rdf_1_1_concepts_abstract_syntax
n_w3c_rdf_schema_rdfs_standard
n_w3c_shacl_core_validation_engine
n_w3c_shacl_shapes_constraint_language
n_w3c_shacl_sparql_target_constraint
n_w3c_sparql_1_1_query_language
n_w3c_trig_rdf_dataset_syntax
n_w3c_turtle_rdf_terse_syntax
n_zero_telemetry_isolation
n_zero_trust_graph_access_control

## First 80 missing pedagogy

id                                     type         missing           file
--                                     ----         -------           ----
n_sle_v5                               technology   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0001.json
n_khawrizm_build                       architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0001.json
n_entity_resolution_protocol           algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0001.json
n_skg_schema                           standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0001.json
n_zero_cloud_op                        constraint   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0001.json
n_composite_scoring                    algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0002.json
n_arabic_morph_scoring                 algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0002.json
n_skg_audit_metrics                    standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0002.json
n_skg_completeness                     requirement  {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0002.json
n_skg_schema_compliance                requirement  {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0002.json
n_skg_contradiction_index              constraint   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0002.json
n_node_model_spec                      standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0003.json
n_edge_model_spec                      standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0003.json
n_action_thresholds                    algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0003.json
n_stable_semantic_ids                  practice     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0003.json
n_provenance_coverage                  requirement  {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0003.json
n_ring0_execution_environment          architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0004.json
n_edge_policy_candidate_validate       protocol     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0004.json
n_bilingual_trigger_eval               component    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0004.json
n_contradiction_index_calc             algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0004.json
n_inference_policy_model_assisted      standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0004.json
n_continuation_protocol                protocol     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0005.json
n_deterministic_sovereignty            concept      {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0005.json
n_graph_reconciliation                 practice     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0005.json
n_semantic_lexical_similarity          algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0005.json
n_quality_audit_status                 decision     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0005.json
n_vector_index_hnsw                    algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0006.json
n_property_graph_model                 architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0006.json
n_graph_transaction_manager            component    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0006.json
n_triplet_loss_metric                  algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0006.json
n_schema_validation_engine             component    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0006.json
n_levenshtein_jaro_distance            algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0007.json
n_arabic_stemming_khoja                algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0007.json
n_deterministic_deduplication          practice     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0007.json
n_graph_traversal_bfs                  algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0007.json
n_zero_telemetry_isolation             architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0007.json
n_merkle_provenance_tree               architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0008.json
n_type_compatibility_matrix            standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0008.json
n_acid_wal_persistence                 technology   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0008.json
n_graph_contradiction_resolver         algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0008.json
n_canonical_semantic_id_generator      algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0008.json
n_shacl_constraint_validation          protocol     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0009.json
n_rdf_owl_ontology_mapper              technology   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0009.json
n_cypher_query_compiler                language     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0009.json
n_zero_knowledge_audit_trail           architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0009.json
n_entity_resolution_benchmark          practice     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0009.json
n_graph_rag_retrieval                  architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0010.json
n_graph_neural_networks                algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0010.json
n_bitemporal_graph_versioning          protocol     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0010.json
n_vf2_subgraph_isomorphism             algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0010.json
n_differential_privacy_graph_masking   practice     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0010.json
n_graph_partitioning_fennel            algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0011.json
n_sparql_11_entailment                 standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0011.json
n_zero_knowledge_graph_traversal       protocol     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0011.json
n_page_rank_personalized               algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0011.json
n_property_graph_schema_shacl_bridge   component    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0011.json
n_hnsw_pq_compression                  algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0012.json
n_temporal_graph_snapshot              concept      {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0012.json
n_cross_lingual_entity_alignment       algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0012.json
n_graph_schema_evolution               practice     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0012.json
n_cryptographic_state_root             architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0012.json
n_raft_consensus_graph_state           protocol     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0013.json
n_graph_embedding_node2vec             algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0013.json
n_owl2_direct_semantics                standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0013.json
n_zero_trust_graph_access_control      architecture {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0013.json
n_named_entity_disambiguation_nel      algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0013.json
n_fpga_graph_traversal_accelerator     technology   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0014.json
n_graph_representation_learning_transe algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0014.json
n_iso_iec_39788_graph_query            standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0014.json
n_federated_graph_entity_resolution    practice     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0014.json
n_bloom_filter_edge_pruning            component    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0014.json
n_cugraph_gpu_accelerator              technology   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0015.json
n_scann_vector_indexing                algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0015.json
n_rdf_star_triple_annotations          standard     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0015.json
n_laplace_privacy_mechanism            protocol     {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0015.json
n_louvain_community_detection          algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0015.json
n_numa_memory_affinity_allocator       technology   {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0016.json
n_roaring_bitmap_index                 algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0016.json
n_graph_rag_triples_extractor          component    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0016.json
n_arabic_morphological_root_qutrub     algorithm    {L0, L1, L2, L3…} chunks\khawrizm_graph_chunk_0016.json


