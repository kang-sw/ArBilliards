cmake_minimum_required		(VERSION		3.1 )
project						(physcore			)

aux_source_directory		("src"								PHYSCORE_SOURCE)
add_library					(physcore		STATIC				${PHYSCORE_SOURCE})

# FINDS NVIDIA PHYSX
set							(PhysX_INCLUDE_DIR					PhysX_INCLUDE_DIR_NOTFOUND 
																CACHE PATH " ")
set							(PhysX_LIBRARY_DIR					PhysX_LIBRARY_DIR_NOTFOUND 
																CACHE PATH " ")

set							(PhysX_LIBRARIES					PhysXExtensions_static_64
																PhysX_64
																PhysXPvdSDK_static_64
																PhysXVehicle_static_64
																PhysXCharacterKinematic_static_64
																PhysXCooking_64
																PhysXCommon_64
																PhysXFoundation_64)

message(${PhysX_LIBRARIES})

# 
target_compile_features		(physcore		PUBLIC				cxx_std_20)
target_include_directories	(physcore		PUBLIC				"${CMAKE_CURRENT_SOURCE_DIR}/include"
											PRIVATE				"${PhysX_INCLUDE_DIR}/")
target_link_directories		(physcore		PRIVATE				"${PhysX_LIBRARY_DIR}/")
target_link_libraries		(physcore		PRIVATE				"${PhysX_LIBRARIES}")

add_library					(physcore::lib  ALIAS				physcore)

# Tiny sample program
aux_source_directory		("sample"							PHYSCORE_TINY_EXEC_SOURCE)
add_executable				(tiny_exec							${PHYSCORE_TINY_EXEC_SOURCE})
add_dependencies			(tiny_exec							physcore)


target_compile_features		(tiny_exec		PUBLIC				cxx_std_20)
target_include_directories	(tiny_exec		PUBLIC				"${CMAKE_CURRENT_SOURCE_DIR}/include"
											PRIVATE				"${PhysX_INCLUDE_DIR}/")
target_link_directories		(tiny_exec		PRIVATE				"${PhysX_LIBRARY_DIR}/")
target_link_libraries		(tiny_exec		PRIVATE				${PhysX_LIBRARIES}
																physcore::lib)

add_executable				(physcore::internals::tiny_exec		
											ALIAS				tiny_exec)