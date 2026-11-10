#!/usr/bin/env python3
"""DiscoveryCache helpers for StoiAgent / CASP17.

PubMed URLs are resolved using structured NCBI title/abstract data by default;
legacy full-text retrieval is available only with pubmed_content_mode="fulltext".
See the accompanying README for summary-cache invalidation instructions.
"""

import os
import re
import json
from hashlib import md5
from typing import Tuple, Optional

import requests
from bs4 import BeautifulSoup



# Define the query
PDB_QUERY = """
        query structure($id: String!) {
        entry(entry_id: $id) {
            rcsb_id
            entry {
            id
            }
            database_2 {
            database_id
            pdbx_database_accession
            }
            rcsb_entry_container_identifiers {
            entity_ids
            pubmed_id
            emdb_ids
            assembly_ids
            }
            rcsb_associated_holdings {
            rcsb_repository_holdings_current_entry_container_identifiers {
                assembly_ids
            }
            rcsb_repository_holdings_current {
                repository_content_types
            }
            }
            rcsb_comp_model_provenance {
            source_url
            source_pae_url
            entry_id
            source_db
            }
            pdbx_database_status {
            pdb_format_compatible
            }
            struct {
            title
            }
            rcsb_ma_qa_metric_global {
            model_id
            ma_qa_metric_global {
                name
                value
                type
                description
                type_other_details
            }
            }
            rcsb_primary_citation {
            id
            pdbx_database_id_DOI
            }
            pubmed {
            rcsb_pubmed_container_identifiers {
                pubmed_id
            }
            rcsb_pubmed_central_id
            rcsb_pubmed_doi
            rcsb_pubmed_abstract_text
            rcsb_pubmed_affiliation_info
            }
            pdbx_deposit_group {
            group_id
            group_type
            }
            rcsb_external_references {
            id
            type
            link
            }
            pdbx_database_PDB_obs_spr {
            id
            replace_pdb_id
            }
            struct_keywords {
            pdbx_keywords
            text
            }
            exptl {
            method
            }
            cell {
            length_a
            length_b
            length_c
            angle_alpha
            angle_beta
            angle_gamma
            }
            symmetry {
            space_group_name_H_M
            }
            software {
            classification
            name
            }
            rcsb_accession_info {
            deposit_date
            initial_release_date
            major_revision
            minor_revision
            }
            pdbx_audit_revision_history {
            ordinal
            data_content_type
            major_revision
            minor_revision
            revision_date
            }
            pdbx_audit_revision_details {
            ordinal
            revision_ordinal
            data_content_type
            type
            description
            }
            pdbx_audit_revision_group {
            ordinal
            revision_ordinal
            data_content_type
            group
            }
            pdbx_database_related {
            content_type
            db_id
            details
            }
            audit_author {
            name
            }
            pdbx_audit_support {
            funding_organization
            country
            grant_number
            ordinal
            }
            pdbx_initial_refinement_model {
            type
            }
            refine {
            pdbx_refine_id
            ls_d_res_high
            ls_R_factor_R_work
            ls_R_factor_R_free
            ls_R_factor_obs
            }
            pdbx_vrpt_summary_diffraction {
            DCC_R
            DCC_Rfree
            }
            pdbx_nmr_ensemble {
            conformers_calculated_total_number
            conformers_submitted_total_number
            conformer_selection_criteria
            }
            em_experiment {
            aggregation_state
            reconstruction_method
            }
            em_3d_reconstruction {
            resolution
            }
            em_software {
            category
            name
            version
            }
            citation {
            id
            title
            rcsb_journal_abbrev
            rcsb_authors
            year
            journal_volume
            page_first
            page_last
            pdbx_database_id_PubMed
            pdbx_database_id_DOI
            }
            pdbx_database_related {
            db_id
            db_name
            }
            ihm_entry_collection_mapping {
            collection_id
            }
            rcsb_entry_info {
            structure_determination_methodology
            ihm_multi_scale_flag
            ihm_multi_state_flag
            ihm_ordered_state_flag
            deposited_model_count
            representative_model
            molecular_weight
            deposited_atom_count
            deposited_polymer_monomer_count
            deposited_modeled_polymer_monomer_count
            deposited_unmodeled_polymer_monomer_count
            polymer_entity_count_protein
            polymer_entity_count_nucleic_acid
            polymer_entity_count_nucleic_acid_hybrid
            }
            rcsb_entry_group_membership {
            group_id
            aggregation_method
            }
            rcsb_binding_affinity {
            comp_id
            type
            value
            unit
            reference_sequence_identity
            provenance_code
            link
            }
            branched_entities {
            rcsb_id
            rcsb_branched_entity_container_identifiers {
                entry_id
                entity_id
                asym_ids
                auth_asym_ids
                reference_identifiers {
                provenance_source
                resource_name
                resource_accession
                }
            }
            prd {
                rcsb_id
                pdbx_reference_molecule {
                prd_id
                chem_comp_id
                name
                type
                class
                }
            }
            rcsb_branched_entity {
                pdbx_description
                formula_weight
            }
            pdbx_entity_branch {
                rcsb_branched_component_count
            }
            branched_entity_instances {
                rcsb_branched_entity_instance_container_identifiers {
                entry_id
                entity_id
                asym_id
                auth_asym_id
                }
                rcsb_branched_struct_conn {
                connect_type
                role
                ordinal_id
                connect_partner {
                    label_asym_id
                    label_seq_id
                    label_comp_id
                }
                connect_target {
                    auth_seq_id
                    label_asym_id
                    label_comp_id
                }
                }
                rcsb_branched_instance_feature {
                name
                type
                feature_value {
                    comp_id
                    details
                }
                }
            }
            }
            polymer_entities {
            polymer_entity_instances {
                rcsb_polymer_entity_instance_container_identifiers {
                auth_asym_id
                asym_id
                entry_id
                entity_id
                }
            }
            rcsb_polymer_entity_container_identifiers {
                entry_id
                entity_id
                asym_ids
                auth_asym_ids
                uniprot_ids
                reference_sequence_identifiers {
                database_accession
                }
            }
            uniprots {
                rcsb_id
                rcsb_uniprot_protein {
                source_organism {
                    scientific_name
                }
                }
                rcsb_uniprot_external_reference {
                reference_name
                reference_id
                }
            }
            rcsb_polymer_entity {
                pdbx_description
                rcsb_ec_lineage {
                id
                }
                pdbx_ec
                rcsb_enzyme_class_combined {
                ec
                provenance_source
                }
            }
            rcsb_polymer_entity_annotation {
                type
                annotation_lineage {
                name
                depth
                }
            }
            rcsb_polymer_entity_group_membership {
                group_id
                similarity_cutoff
                aggregation_method
            }
            entity_poly {
                type
                rcsb_entity_polymer_type
                pdbx_seq_one_letter_code_can
                rcsb_sample_sequence_length
                rcsb_mutation_count
            }
            rcsb_entity_source_organism {
                scientific_name
                ncbi_scientific_name
                rcsb_gene_name {
                value
                provenance_source
                }
            }
            rcsb_entity_host_organism {
                ncbi_scientific_name
            }
            prd {
                rcsb_id
                pdbx_reference_molecule {
                prd_id
                chem_comp_id
                name
                type
                class
                }
            }
            chem_comp_nstd_monomers {
                chem_comp {
                id
                name
                formula
                type
                mon_nstd_parent_comp_id
                }
            }
            polymer_entity_instances {
                rcsb_polymer_instance_annotation {
                type
                annotation_id
                }
                rcsb_polymer_struct_conn {
                role
                connect_type
                connect_partner {
                    label_asym_id
                }
                connect_target {
                    label_asym_id
                }
                }
            }
            }
            nonpolymer_entities {
            rcsb_nonpolymer_entity_container_identifiers {
                entry_id
                entity_id
                auth_asym_ids
                asym_ids
                nonpolymer_comp_id
            }
            rcsb_nonpolymer_entity_annotation {
                type
            }
            nonpolymer_entity_instances {
                rcsb_nonpolymer_entity_instance_container_identifiers {
                auth_seq_id
                auth_asym_id
                asym_id
                entry_id
                entity_id
                }
                rcsb_nonpolymer_instance_validation_score {
                ranking_model_fit
                ranking_model_geometry
                average_occupancy
                is_subject_of_investigation
                is_subject_of_investigation_provenance
                }
            }
            rcsb_nonpolymer_entity {
                pdbx_description
            }
            nonpolymer_comp {
                chem_comp {
                id
                formula_weight
                name
                formula
                }
                pdbx_reference_molecule {
                prd_id
                chem_comp_id
                type
                name
                class
                }
                rcsb_chem_comp_descriptor {
                InChIKey
                }
            }
            }
            assemblies {
            rcsb_assembly_container_identifiers {
                assembly_id
            }
            pdbx_struct_assembly {
                id
                details
                method_details
                oligomeric_details
                oligomeric_count
            }
            pdbx_struct_assembly_auth_evidence {
                experimental_support
            }
            rcsb_struct_symmetry {
                kind
                type
                symbol
                oligomeric_state
                stoichiometry
            }
            rcsb_assembly_info {
                modeled_polymer_monomer_count
            }
            }
        }
        }
    """


#!/usr/bin/env python3
import xml.etree.ElementTree as ET

def extract_abstract_from_xml(xml_text: str, pmid: str) -> str:
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is not None and pmid_node.text == pmid:
                abs_nodes = article.findall(".//AbstractText")
                if abs_nodes:
                    return f"PMID {pmid}: " + " ".join(n.text for n in abs_nodes if n.text)
        return f"PMID {pmid}: (no abstract found)"
    except Exception as e:
        return f"PMID {pmid}: error {e}"

class DiscoveryCache:
    """Cache-first discovery and summarization for RCSB, UniProt, PubMed, and articles."""

    MODEL_CTX = 131_072
    GEN_MARGIN = 1024  # safety buffer for new tokens + overhead

    def __init__(self,
                 rcsb_cache="pdb_json_cache",
                 uniprot_cache="uniprot_cache",
                 pubmed_cache="pubmed_cache",
                 article_cache="fulltext_cache",
                 pubmed_content_mode="abstract"):
        """Initialize the cache.

        pubmed_content_mode="abstract" (default) retrieves only the
        structured title and abstract from NCBI EFetch PubMed XML. It never
        falls back to full-text or a current publisher/PMC web page.

        Use pubmed_content_mode="fulltext" explicitly to opt back into the
        legacy Europe PMC / PMC retrieval behavior for other projects.
        """
        if pubmed_content_mode not in {"abstract", "fulltext"}:
            raise ValueError("pubmed_content_mode must be 'abstract' or 'fulltext'")
        self.pubmed_content_mode = pubmed_content_mode
        self.rcsb_cache = rcsb_cache
        self.uniprot_cache = uniprot_cache
        self.pubmed_cache = pubmed_cache
        self.article_cache = article_cache
        for d in [rcsb_cache, uniprot_cache, pubmed_cache, article_cache]:
            os.makedirs(d, exist_ok=True)

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _safe_key(text: str, max_length=80) -> str:
        key = re.sub(r'[^A-Za-z0-9._-]', '_', text)
        if len(key) > max_length:
            key = key[:max_length] + "_" + md5(text.encode("utf-8")).hexdigest()[:8]
        return key

    # -----------------------
    # RCSB PDB JSON
    # -----------------------
    def get_rcsb_json(self, pdb_id: str) -> dict:
        """
        Load full PDB JSON from cache if available, else fetch via GraphQL API.
        """
        path = os.path.join(self.rcsb_cache, f"{self._safe_key(pdb_id)}.json")

        # 1. Cached
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 2. GraphQL fetch
        url = "https://data.rcsb.org/graphql"

        variables = {"id": pdb_id}
        try:
            response = requests.post(url, json={"query": PDB_QUERY, "variables": variables}, timeout=20)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return {"error": f"Could not fetch {pdb_id} GraphQL: {e}"}

        # 3. Save to cache
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data


    def summarize_rcsb(self, pdb_id: str) -> str:
        """
        Summarize deposited PDB assembly information only.

        Historical eligibility is verified separately by
        ResearchToolExecutor._pdb_allowed() before this function
        is called.

        Exclude RCSB-computed symmetry and stoichiometry
        annotations without independent historical timestamps.
        """
        root = self.get_rcsb_json(pdb_id)

        if "error" in root:
            return root["error"]

        entry = (root.get("data") or {}).get("entry") or {}

        if not entry:
            return f"PDB code not valid: {pdb_id}"

        bits = []

        struct = entry.get("struct") or {}
        title = struct.get("title")
        if title:
            bits.append(f"Structure title: {title}")

        for asm in entry.get("assemblies") or []:
            deposited = asm.get("pdbx_struct_assembly") or {}

            asm_id = deposited.get("id")
            if not asm_id:
                asm_id = (
                    asm.get("rcsb_assembly_container_identifiers") or {}
                ).get("assembly_id", "unknown")

            count = deposited.get("oligomeric_count")
            details = deposited.get("oligomeric_details")
            method = deposited.get("method_details")

            if count is not None:
                bits.append(
                    f"Assembly {asm_id}: "
                    f"deposited oligomeric count={count}"
                )

            if details:
                bits.append(
                    f"Assembly {asm_id}: "
                    f"deposited oligomeric state={details}"
                )

            if method:
                bits.append(
                    f"Assembly {asm_id}: method={method}"
                )

        if not bits:
            return f"No usable deposited assembly evidence for {pdb_id}."

        return "\n".join(bits)


    def collect_uniprot_ids(self, pdb_id: str, limit: int = 3) -> list:
        """Extract up to `limit` UniProt IDs from cached RCSB JSON."""
        root = self.get_rcsb_json(pdb_id)
        if "error" in root:
            return []

        entry = root.get("data", {}).get("entry", {})
        entities = entry.get("polymer_entities", [])

        uids = set()
        for e in entities:
            ids = e.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids")
            if ids and isinstance(ids, list):
                uids.update(ids)
            if len(uids) >= limit:
                break
        return list(uids)[:limit]

    def build_pubmed_search_query(self, template_id: str, pdb_data: dict = None) -> str:
        """
        Construct a more discriminative PubMed query for a template PDB entry.
        """
        terms = []

        root = self.get_rcsb_json(template_id)
        if "error" in root:
            return root["error"]

        pdb_data = root.get("data", {})
        # Always include the PDB ID
        # terms.append(f"{template_id}[Title/Abstract]")

        # Add protein description or keywords if available
        if pdb_data:
            title = pdb_data.get("entry", {}).get("struct", {}).get("title", "")
            if title:
                terms.append(title)
            keywords = pdb_data.get("entry", {}).get("struct_keywords", {}).get("text", "")
            if keywords:
                terms.append(keywords)

        # Add stoichiometry/oligomer keywords
        terms.append("(stoichiometry OR oligomer OR assembly OR complex)")

        return " ".join(terms)


    # -----------------------
    # UniProt
    # -----------------------
    def get_uniprot_json(self, uid: str) -> dict:
        """
        Load full UniProt JSON from cache if available, else fetch from UniProt API.
        """
        path = os.path.join(self.uniprot_cache, f"{self._safe_key(uid)}.json")

        # 1. Cached
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 2. Fetch live
        url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return {"error": f"Could not fetch UniProt {uid}: {e}"}

        # 3. Save to cache
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data

    def get_uniprot_observation(self, uid: str) -> str:
        """
        Summarize UniProt entry from cached JSON.
        Includes protein name, organism, subunit annotation (stoichiometry),
        and functional information if available.
        """
        root = self.get_uniprot_json(uid)
        if not root or "error" in root:
            return f"UniProt: {uid}. Error fetching data."

        # Protein name
        protein_name = (
            root.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value")
        )

        # Organism
        organism = root.get("organism", {}).get("scientificName")

        # Initialize placeholders
        subunit_info, function_info = None, None

        # Parse comments for subunit and function
        for comment in root.get("comments", []):
            ctype = comment.get("commentType")
            if ctype == "SUBUNIT" and not subunit_info:
                subunit_info = comment.get("texts", [{}])[0].get("value")
            elif ctype == "FUNCTION" and not function_info:
                function_info = comment.get("texts", [{}])[0].get("value")

        # Build output
        parts = [f"UniProt: {uid}."]
        if protein_name:
            parts.append(f"Protein name: {protein_name}.")
        if organism:
            parts.append(f"Organism: {organism}.")
        if subunit_info:
            parts.append(f"Subunit: {subunit_info}")
        if function_info:
            parts.append(f"Function: {function_info}")
        if len(parts) == 1:
            parts.append("No detailed description available.")

        return " ".join(parts)

    def get_pubmed_search_observation(self, query: str, max_results=3, read_articles=False) -> str:
        """PubMed search with cache-first. Optionally fetch abstracts."""
        path = os.path.join(self.pubmed_cache, f"{self._safe_key(query)}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        try:
            r = requests.get(url, params=params, timeout=15).json()
            ids = r.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            ids, obs = [], f"PubMed search for \"{query}\" failed: {e}"

        if not ids:
            obs = f"PubMed search for \"{query}\": no hits found."
        else:
            if read_articles:
                abs_path = os.path.join(self.pubmed_cache, f"PMIDS_{'_'.join(ids)}.txt")
                if os.path.exists(abs_path):
                    with open(abs_path, "r", encoding="utf-8") as f:
                        abstracts = f.read().strip()
                else:
                    # 🟢 Batch efetch call for multiple PMIDs at once
                    abs_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    abs_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
                    try:
                        abs_r = requests.get(abs_url, params=abs_params, timeout=15)
                        abs_r.raise_for_status()
                        abs_xml = abs_r.text
                        # Extract abstracts
                        abstracts = []
                        for pmid in ids:
                            abstracts.append(extract_abstract_from_xml(abs_xml, pmid))
                        abstracts = "\n\n".join(abstracts)

                        # Save raw XML
                        with open(abs_path, "w", encoding="utf-8") as f:
                            f.write(abstracts)

                    except Exception as e:
                        abstracts = f"Error fetching PMIDs {ids}: {e}"

                obs = f"PubMed search for \"{query}\" returned {len(ids)} hits: {', '.join(ids)}.\n\n" + abstracts
            else:
                obs = f"PubMed search for \"{query}\": found {len(ids)} hits. Example PMID(s): {', '.join(ids)}."

        with open(path, "w", encoding="utf-8") as f:
            f.write(obs)
        return obs


    # -----------------------
    # Summarization
    # -----------------------
    def _tok_len(self, tokenizer, s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0])

    def _split_by_tokens(self, tokenizer, text: str, max_tokens: int, overlap: int = 128):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        if max_tokens <= 0:
            max_tokens = 2048
        chunks, step = [], max(1, max_tokens - overlap)
        for i in range(0, len(ids), step):
            chunk_ids = ids[i:i + max_tokens]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        return chunks

    # def _build_sum_prompt(self, text: str, max_words: int) -> str:
    #     return (
    #         f"You are extracting oligomeric state / stoichiometry evidence from scientific text.\n"
    #         f"Write at most {max_words} words.\n\n"
    #         "Output format (use exactly these field names):\n"
    #         "Stoichiometry: <e.g., A2B2, 2:2, homodimer, heterotetramer, unknown>\n"
    #         "Complex/Protein: <name(s) if present>\n"
    #         "Evidence: <one short sentence from the text that supports the stoichiometry; copy verbatim if possible>\n"
    #         "Condition/Assay: <optional; e.g., SEC-MALS, native MS, cryo-EM, X-ray, in solution; or 'not stated'>\n"
    #         "Confidence: <high/medium/low based on how explicit the text is>\n\n"
    #         "Rules:\n"
    #         "- If the text does NOT explicitly state stoichiometry/oligomeric state, write Stoichiometry: unknown.\n"
    #         "- Prefer explicit numbers/terms (dimer/trimer/tetramer, A2B2, 2:1) over vague wording.\n"
    #         "- Do not invent a stoichiometry.\n"
    #         "- Keep Evidence short (<= 25 words).\n\n"
    #         f"Text:\n{text}\n"
    #     )

    def _build_sum_prompt(self, text: str, max_words: int) -> str:
        return (
            f"Summarize the following scientific text in at most {max_words} words.\n\n"
            "Guidelines:\n"
            "- State the main scientific finding(s).\n"
            "- Mention the protein/complex name if given.\n"
            "- Extract any stoichiometry or oligomeric state (e.g., 'A2B2C4').\n"
            "- Ignore experimental methods and mutations.\n"
            "- Output only the summary sentence(s).\n\n"
            f"Full text:\n{text}\n\nSummary:"
        )

    def _safe_summary_call(self, llm, text: str, max_words: int, gen_tokens: int):
        def clean_reply(text: str) -> str:
            bad_tokens = ["<end_of_turn>", "<bos>", "<eos>", "<|end_of_text|>"]
            for t in bad_tokens:
                text = text.replace(t, "")
            return text.strip()

        tokenizer, pipe = llm.tokenizer, llm.pipe

        # ✅ Single source of truth: always use llm.gen_kwargs
        gen_kwargs = getattr(llm, "gen_kwargs", {}).copy()
        # Enforce a couple of things that should never vary
        gen_kwargs["return_full_text"] = False
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        # (optional) also enforce pad token id
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        # Model-specific safe input lengths
        if "gemma" in getattr(llm, "model_key", "").lower():
            MAX_SAFE_INPUT_TOKENS = self.MODEL_CTX // 2
        else:
            MAX_SAFE_INPUT_TOKENS = self.MODEL_CTX

        # Budget estimate
        header = self._build_sum_prompt("", max_words)
        header_tokens = self._tok_len(tokenizer, header)
        budget_for_body = self.MODEL_CTX - self.GEN_MARGIN - gen_tokens - header_tokens
        if budget_for_body <= 512:
            budget_for_body = 2048

        token_len = self._tok_len(tokenizer, text)

        # -------------------------
        # Chunk/map-reduce path
        # -------------------------
        if token_len > min(budget_for_body, MAX_SAFE_INPUT_TOKENS):
            print(f"⚠️ Forcing chunk mode: {token_len} tokens > cutoff {MAX_SAFE_INPUT_TOKENS}")
            chunks = self._split_by_tokens(tokenizer, text, MAX_SAFE_INPUT_TOKENS, overlap=256)
            partial_summaries = []

            for ch in chunks:
                prompt = self._build_sum_prompt(ch, max_words=max_words // 2)

                # Gemma chat wrapping if needed
                if "gemma" in getattr(llm, "model_key", "").lower():
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                    )

                out = pipe(
                    prompt,
                    max_new_tokens=max(64, gen_tokens // 2),
                    **gen_kwargs
                )
                part = clean_reply(out[0]["generated_text"])
                partial_summaries.append(part)

            combined = "\n".join(partial_summaries)
            return self._safe_summary_call(llm, combined, max_words, max(64, gen_tokens // 2))

        # -------------------------
        # Single-shot path
        # -------------------------
        prompt = self._build_sum_prompt(text, max_words)
        if "gemma" in getattr(llm, "model_key", "").lower():
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

        try:
            out = pipe(
                prompt,
                max_new_tokens=gen_tokens,
                **gen_kwargs
            )
            gen = clean_reply(out[0]["generated_text"])
            words = gen.split()
            return (" ".join(words[:max_words]) + "...") if len(words) > max_words else gen

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and gen_tokens > 64:
                print("⚠️ OOM detected, retrying with fewer gen_tokens")
                return self._safe_summary_call(llm, text, max_words, gen_tokens // 2)
            raise

    def summarize_text(self, citation: str, text: str, llm, max_words=250) -> str:
        """Cache-first summarization: load if available, else generate and save."""
        safe_key = self._safe_key(citation)
        folder = os.path.join(self.article_cache, safe_key)
        os.makedirs(folder, exist_ok=True)
        summary_path = os.path.join(folder, "summary.txt")
        #print(f"Regenerating summary for {citation}")
        #os.system(f"rm {summary_path}")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return f.read().strip()

        print(summary_path)
        gen_tokens = min(768, max(128, int(max_words * 2)))
        summary = self._safe_summary_call(llm, text, max_words, gen_tokens)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary.strip())

        return summary.strip()

        # -----------------------
    # PubMed + Article Fetch
    # -----------------------
    def extract_text_from_pmc_xml(self, xml_content: str) -> str:
        """Extract title, abstract, and body from PMC JATS/OAI XML."""
        soup = BeautifulSoup(xml_content, "xml")

        dc = soup.find("oai_dc:dc")
        if dc:
            title = dc.find("dc:title")
            abstract = dc.find("dc:description")
            creators = [c.get_text(strip=True) for c in dc.find_all("dc:creator")]
            parts = []
            if title:
                parts.append(f"Title: {title.get_text(strip=True)}")
            if abstract:
                parts.append("Abstract: " + abstract.get_text(" ", strip=True))
            if creators:
                parts.append("Authors: " + ", ".join(creators))
            return "\n\n".join(parts).strip()

        article = soup.find("article")
        if article:
            soup = article

        text_parts = []
        title = soup.find("article-title")
        if title:
            text_parts.append(f"Title: {title.get_text(strip=True)}")

        abstract = soup.find("abstract")
        if abstract:
            abs_paras = [p.get_text(" ", strip=True) for p in abstract.find_all("p")]
            if abs_paras:
                text_parts.append("Abstract: " + " ".join(abs_paras))

        body = soup.find("body")
        if body:
            body_parts = []
            for sec in body.find_all("sec"):
                sec_title = sec.find("title")
                sec_paras = [p.get_text(" ", strip=True) for p in sec.find_all("p")]
                if sec_title:
                    body_parts.append(f"## {sec_title.get_text(strip=True)}")
                if sec_paras:
                    body_parts.extend(sec_paras)
            if body_parts:
                text_parts.append("Body:\n" + "\n\n".join(body_parts))

        return "\n\n".join(text_parts).strip()

    # def fetch_doi_content(self, doi: str, headers: dict) -> Tuple[Optional[str], str, list, str]:
    #     """Fetch article text for a DOI using EuropePMC, PMC, or CrossRef fallback."""
    #     text, status, trace = None, "error", []

    #     #changed to core?
    #     url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&resulttype=fulltext&format=json"
    #     # print(url)
    #     try:
    #         r = requests.get(url, headers=headers, timeout=15).json()
    #         hits = r.get("resultList", {}).get("result", [])
    #         if hits and hits[0].get("fullText"):
    #             return f"Title: {hits[0].get('title','')}\n\nBody: {hits[0]['fullText']}", "fulltext", trace, ""
    #     except Exception as e:
    #         trace.append({"step": "EuropePMC fulltext", "error": str(e)})

    #     return None, "error", trace, "DOI fetch failed"

    def fetch_doi_content(self, doi: str, headers: dict) -> Tuple[Optional[str], str, list, str]:
        """
        Try EuropePMC full text first; if unavailable, follow DOI redirect
        and attempt to extract limited text or detect PDF/paywall.
        """
        text, status, trace, final_reason = None, "error", [], ""

        # --- 1. Try EuropePMC API ---
        epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&resulttype=fulltext&format=json"
        trace.append({"step": "EuropePMC", "url": epmc_url})
        try:
            r = requests.get(epmc_url, headers=headers, timeout=15).json()
            hits = r.get("resultList", {}).get("result", [])
            if hits and hits[0].get("fullText"):
                text = f"Title: {hits[0].get('title','')}\n\nBody: {hits[0]['fullText']}"
                return text, "fulltext", trace, "EuropePMC fulltext retrieved"
        except Exception as e:
            trace.append({"step": "EuropePMC error", "error": str(e)})

        # --- 2. Follow DOI redirect ---
        doi_url = f"https://doi.org/{doi}"
        trace.append({"step": "DOI redirect", "url": doi_url})
        try:
            r = requests.get(doi_url, headers=headers, allow_redirects=True, timeout=15)
            trace.append({
                "final_url": r.url,
                "status": r.status_code,
                "content_type": r.headers.get("Content-Type", "")
            })

            if r.status_code != 200:
                final_reason = f"HTTP {r.status_code}"
                return None, "error", trace, final_reason

            ctype = r.headers.get("Content-Type", "").lower()
            if "pdf" in ctype:
                text = f"[PDF detected but not downloaded] {r.url}"
                return text, "pdf", trace, "PDF detected (likely paywalled)"

            if "html" in ctype:
                soup = BeautifulSoup(r.text, "html.parser")
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
                text = "\n".join(paragraphs[:100])
                if len(text.strip()) < 50:
                    final_reason = "HTML too short or no body text"
                    return None, "error", trace, final_reason
                return text, "html", trace, "HTML extracted from publisher page"

            final_reason = f"Unsupported content type {ctype}"
            return None, "error", trace, final_reason

        except requests.exceptions.RequestException as e:
            trace.append({"step": "DOI fetch exception", "error": str(e)})
            final_reason = str(e)

        return text, status, trace, final_reason


    def _fetch_pubmed_abstract(self, pmid: str, headers: dict) -> Tuple[Optional[str], str, list, str]:
        """Retrieve ONLY the title and abstract of the exact requested PMID.

        Use structured NCBI PubMed EFetch XML; never use Europe PMC fulltext,
        PMC article body, publisher HTML, or other fallback sources.
        Publication-date eligibility is checked by ResearchToolExecutor before
        this method is called in the CASP17 tool path.
        """
        pmid = str(pmid).strip()
        if not re.fullmatch(r"\d{5,10}", pmid):
            return None, "error", [], "Invalid PubMed ID"

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        trace = [{"step": "NCBI EFetch title/abstract only", "url": url, "pmid": pmid}]
        try:
            response = requests.get(
                url,
                params={"db": "pubmed", "id": pmid, "retmode": "xml"},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except Exception as e:
            trace[-1]["error"] = str(e)
            return None, "error", trace, f"PubMed EFetch failed: {e}"

        def text_content(node):
            if node is None:
                return ""
            return " ".join(" ".join(node.itertext()).split())

        for record in root.findall(".//PubmedArticle"):
            id_node = record.find("./MedlineCitation/PMID")
            if id_node is None or (id_node.text or "").strip() != pmid:
                continue

            article = record.find("./MedlineCitation/Article")
            if article is None:
                return None, "error", trace, "Matching PMID has no Article element"

            title = text_content(article.find("./ArticleTitle"))
            abstract = article.find("./Abstract")
            parts = []
            if abstract is not None:
                for section in abstract.findall("./AbstractText"):
                    body = text_content(section)
                    if not body:
                        continue
                    label = (section.get("Label") or "").strip()
                    parts.append(f"{label}: {body}" if label else body)

            # We do not fall back to full text, even for articles without abstracts.
            if not parts:
                trace[-1]["reason"] = "no abstract present"
                return None, "error", trace, "No PubMed abstract available for this PMID"

            observation = f"PMID: {pmid}\n"
            if title:
                observation += f"Title: {title}\n"
            observation += "Abstract: " + "\n".join(parts)
            trace[-1]["success"] = True
            return observation, "abstract", trace, ""

        trace[-1]["reason"] = "PMID not present in EFetch response"
        return None, "error", trace, "Requested PMID missing from PubMed XML"

    def fetch_pubmed_content(self, pmid: str, headers: dict) -> Tuple[Optional[str], str, list, str]:
        """Retrieve an abstract by default, or legacy full text when opted in."""
        if self.pubmed_content_mode == "abstract":
            return self._fetch_pubmed_abstract(pmid, headers)
        return self._fetch_pubmed_fulltext(pmid, headers)

    def _fetch_pubmed_fulltext(self, pmid: str, headers: dict) -> Tuple[Optional[str], str, list, str]:
        """
        Legacy OPTIONAL PubMed retrieval (fulltext → PMC OAI → abstract).
        Returns (text, status, trace, final_reason).
        """
        text, status = None, "error"
        trace = []

        # 1. EuropePMC fulltext
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}&resulttype=fulltext&format=json"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            hits = r.json().get("resultList", {}).get("result", [])
            if hits and hits[0].get("fullText"):
                text = f"Title: {hits[0].get('title','')}\n\nBody: {hits[0]['fullText']}"
                return text, "fulltext", trace + [{"step": "EuropePMC fulltext", "url": url, "success": True}], ""
            trace.append({"step": "EuropePMC fulltext", "url": url, "success": False, "reason": "no fullText"})
        except Exception as e:
            trace.append({"step": "EuropePMC fulltext", "url": url, "success": False, "error": str(e)})

        # 2. PMC OAI
        try:
            idconv_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
            r = requests.get(idconv_url, timeout=10)
            r.raise_for_status()
            records = r.json().get("records", [])
            if records:
                pmc = records[0].get("pmcid")
                if pmc:
                    pmc_num = pmc.replace("PMC", "")
                    for prefix in ["pmc", "oai_dc"]:
                        xml_url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmc_num}&metadataPrefix={prefix}"
                        try:
                            xml_response = requests.get(xml_url, timeout=20)
                            xml_response.raise_for_status()
                            parsed = self.extract_text_from_pmc_xml(xml_response.text).strip()
                            if parsed:
                                notes, status_here = None, "title"
                                if "Body:" in parsed:
                                    status_here, notes = "fulltext", "body"
                                elif "Abstract:" in parsed:
                                    status_here, notes = "abstract", "abstract"
                                elif "Title:" in parsed:
                                    status_here, notes = "title", "title"
                                trace.append({"step": f"PMC OAI ({prefix})", "url": xml_url, "success": True, "notes": notes})
                                return parsed, status_here, trace, ""
                            trace.append({"step": f"PMC OAI ({prefix})", "url": xml_url, "success": False, "reason": "empty parse"})
                        except Exception as e:
                            trace.append({"step": f"PMC OAI ({prefix})", "url": xml_url, "success": False, "error": str(e)})
                else:
                    trace.append({"step": "PMC OAI", "success": False, "reason": "no PMCID available"})
        except Exception as e:
            trace.append({"step": "PMC ID conversion", "success": False, "error": str(e)})

        # 3. EuropePMC abstract
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}&resulttype=core&format=json"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            hits = r.json().get("resultList", {}).get("result", [])
            if hits and hits[0].get("abstractText"):
                text = f"Title: {hits[0].get('title','')}\n\nAbstract: {hits[0]['abstractText']}"
                return text, "abstract", trace + [{"step": "EuropePMC abstract", "url": url, "success": True}], ""
            trace.append({"step": "EuropePMC abstract", "url": url, "success": False, "reason": "no abstractText"})
        except Exception as e:
            trace.append({"step": "EuropePMC abstract", "url": url, "success": False, "error": str(e)})

        # If all failed
        final_reason = "unknown failure"
        if trace:
            last = trace[-1]
            if "error" in last:
                final_reason = last["error"]
            elif "reason" in last:
                final_reason = last["reason"]
            else:
                final_reason = f"last step failed: {last['step']}"
        return None, "error", trace, final_reason

    def fetch_and_cache(self, citation: str) -> Tuple[str, str]:
        """Cache-first fetch for an article (DOI or PubMed)."""
        citation = citation.strip()
        safe_key = self._safe_key(citation)
        is_pubmed = "pubmed.ncbi.nlm.nih.gov" in citation
        if is_pubmed and self.pubmed_content_mode == "abstract":
            # Never reuse a legacy full-text cache for strict abstract-only mode.
            safe_key += "__pubmed_abstract_v1"
        folder = os.path.join(self.article_cache, safe_key)
        os.makedirs(folder, exist_ok=True)

        cache_file = os.path.join(folder, "content.txt")
        meta_path = os.path.join(folder, "meta.json")

        if os.path.exists(cache_file) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (not is_pubmed or self.pubmed_content_mode != "abstract" or
                    (meta.get("content_mode") == "abstract" and
                     meta.get("status") == "abstract")):
                with open(cache_file, "r", encoding="utf-8") as f:
                    text = f.read()
                return text, citation, meta.get("status", "cached")

        headers = {"User-Agent": "StoichiometryAgent/1.0 (mailto:your_email@example.com)"}
        text, status, trace, final_reason = None, "error", [], "not attempted"

        if citation.startswith("https://doi.org/"):
            doi = citation.replace("https://doi.org/", "")
            text, status, trace, final_reason = self.fetch_doi_content(doi, headers)
        elif "pubmed.ncbi.nlm.nih.gov" in citation:
            match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", citation)
            if match:
                pmid = match.group(1)
                text, status, trace, final_reason = self.fetch_pubmed_content(pmid, headers)
        else:
            trace.append({"step": "parse", "reason": "unsupported citation format"})
            final_reason = "unsupported citation format"

        if status != "error" and text:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
            meta = {"citation": citation, "status": status, "length": len(text),
                    "content_mode": self.pubmed_content_mode if is_pubmed else "doi",
                    "trace": trace}
        else:
            meta = {"citation": citation, "status": "error",
                    "content_mode": self.pubmed_content_mode if is_pubmed else "doi",
                    "trace": trace, "final_reason": final_reason}

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return text, citation, status


