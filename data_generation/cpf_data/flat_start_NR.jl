function flat_start_NR!(net)
    for (_,bus) in net["bus"]
        bus["va"] = 0.0
        if bus["bus_type"] == 1
            bus["vm"] = 1.0
        end
    end
end