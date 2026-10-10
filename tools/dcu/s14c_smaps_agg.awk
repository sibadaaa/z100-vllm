# S14c: split a /proc/<pid>/smaps stream into anonymous-RSS buckets by VMA kind.
#
# Why by-kind matters for S14c: glibc's *main* arena is the `[heap]` (brk) VMA,
# but every *secondary* arena is an unnamed `rw-p` mmap -- so `MALLOC_ARENA_MAX`
# shows up in the "unnamed" bucket, not in `[heap]`.  Splitting the two is what
# tells us whether the glibc arena cap can pay for the missing GB.
#
# Mapping header lines start with the address range; with the default FS the 6th
# field is the mapping name (absent for anonymous mmaps).
#
# Prints: heap_mib unnamed_mib named_anon_mib stack_mib other_anon_mib file_mib anon_total_mib
#   heap/unnamed/named/stack/other are `Anonymous:` sums -> their total == RssAnon;
#   file_mib is the `Rss:` sum over file-backed VMAs -> ~= RssFile.
/^[0-9a-fA-F]+-[0-9a-fA-F]+ / {
  if (NF < 6)                 kind = "unnamed";
  else if ($6 == "[heap]")    kind = "heap";
  else if ($6 == "[stack]")   kind = "stack";
  else if ($6 == "[anon]")    kind = "named";
  else if ($6 ~ /^\[/)        kind = "other";
  else                        kind = "file";
  next
}
/^Rss:/       { rss[kind] += $2; next }
/^Anonymous:/ { anon[kind] += $2; next }
END {
  a = anon["heap"] + anon["unnamed"] + anon["named"] + anon["stack"] + anon["other"];
  printf "%d %d %d %d %d %d %d",
    anon["heap"]/1024, anon["unnamed"]/1024, anon["named"]/1024,
    anon["stack"]/1024, anon["other"]/1024, rss["file"]/1024, a/1024;
}
