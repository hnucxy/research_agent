import streamlit as st


def render_search_settings(
    search_source: str,
    semantic_sort_by: str,
    semantic_year_filter: str,
) -> tuple[str, str, str]:
    with st.expander("搜索设置", expanded=True):
        source_options = {
            "arXiv API": "arxiv",
            "Semantic Scholar API": "semantic_scholar",
        }
        selected_source_label = st.radio(
            "选择文献检索数据源",
            options=list(source_options.keys()),
            index=0 if search_source == "arxiv" else 1,
        )
        search_source = source_options[selected_source_label]
        st.session_state.search_source = search_source

        if search_source == "semantic_scholar":
            semantic_sort_options = {
                "Sort by relevance": "relevance",
                "Sort by citation count": "citation_count",
                "Sort by most influential papers": "most_influential",
                "Sort by recency": "recency",
            }
            current_sort_index = (
                list(semantic_sort_options.values()).index(semantic_sort_by)
                if semantic_sort_by in semantic_sort_options.values()
                else 0
            )
            selected_sort_label = st.selectbox(
                "Semantic Scholar 排序方式",
                options=list(semantic_sort_options.keys()),
                index=current_sort_index,
            )
            semantic_sort_by = semantic_sort_options[selected_sort_label]
            st.session_state.semantic_sort_by = semantic_sort_by
            semantic_year_filter = st.text_input(
                "Semantic Scholar year filter",
                value=semantic_year_filter,
                placeholder="e.g. 2020-2024, 2024, 2021-, -2020",
            ).strip()
            st.session_state.semantic_year_filter = semantic_year_filter
        else:
            semantic_sort_by = st.session_state.get("semantic_sort_by", "relevance")
            semantic_year_filter = st.session_state.get("semantic_year_filter", "")

    return search_source, semantic_sort_by, semantic_year_filter

