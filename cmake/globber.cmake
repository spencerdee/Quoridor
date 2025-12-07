function(globber PATTERNS)
    set(SOURCE_FILES)
    foreach(PATTERN ${PATTERNS})
        set(TEMP)
        file(GLOB_RECURSE TEMP "${PROJECT_SOURCE_DIR}/src/${PATTERN}")
        list(APPEND SOURCE_FILES ${TEMP})
    endforeach(PATTERN ${PATTERNS})

    foreach(SRC ${SOURCE_FILES})
        message(STATUS "GLOBBED: " ${SRC})
    endforeach(SRC ${SOURCE_FILES})

    set(SOURCE_FILES ${SOURCE_FILES} PARENT_SCOPE)
endfunction(globber)